import os
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from svd_head.model import MatrixFactorization

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE    = Path(os.path.dirname(os.path.abspath(__file__))).parent
OUT_DIR = BASE / "training_data"

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_DIM = 64   # must match what was used in train_mf.py

# ── 1. Load train / test CSVs ──────────────────────────────────────────────────
print("[1/3] Loading train/test data...")
train_df = pd.read_csv(OUT_DIR / "train.csv")
test_df  = pd.read_csv(OUT_DIR / "test.csv")
print(f"  train_df : {len(train_df):,} rows")
print(f"  test_df  : {len(test_df):,} rows")

# ── 2. Load MF checkpoint (user2idx, video2idx, model weights) ─────────────────
print("\n[2/3] Loading MF model checkpoint...")
checkpoint = torch.load(OUT_DIR / "mf_model.pt", map_location="cpu", weights_only=False)

user2idx  = checkpoint["user2idx"]
video2idx = checkpoint["video2idx"]
idx2video = checkpoint["idx2video"]
cfg       = checkpoint["config"]
N_USERS   = cfg["n_users"]
N_VIDEOS  = cfg["n_videos"]

print(f"  Users  : {N_USERS:,}")
print(f"  Videos : {N_VIDEOS:,}")


model = MatrixFactorization(N_USERS, N_VIDEOS, EMBEDDING_DIM)
model.load_state_dict(checkpoint["model_state"])
model.eval()
print("  MF model loaded.")

# ── 3. Load candidates ─────────────────────────────────────────────────────────
print("\n[3/3] Loading candidates...")
candidates = pd.read_csv(OUT_DIR / "candidates.csv")
print(f"  {len(candidates):,} rows  ({candidates['user_id'].nunique():,} users)")

# ── 4. Define feature columns ──────────────────────────────────────────────────
drop_cols = [
    'date', 'hourmin', 'time_ms', 'is_click', 'is_like', 'is_follow',
    'is_comment', 'is_forward', 'is_hate', 'long_view', 'play_time_ms',
    'duration_ms', 'profile_stay_time', 'comment_stay_time',
    'is_profile_enter', 'tab',
]
feature_cols = [c for c in train_df.columns
                if c not in drop_cols + ['user_id', 'video_id', 'label']]

cat_cols = ['user_active_degree', 'follow_user_num_range', 'fans_user_num_range',
            'friend_user_num_range', 'register_days_range',
            'video_type', 'upload_type', 'music_type', 'tag']
cat_cols = [c for c in cat_cols if c in feature_cols]
num_cols = [c for c in feature_cols if c not in cat_cols]

print(f"\n  Numeric features     : {len(num_cols)}")
print(f"  Categorical features : {len(cat_cols)}")

# ── 5. Fit label encoders ──────────────────────────────────────────────────────
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    all_vals = pd.concat([train_df[col], test_df[col]]).astype(str).fillna("NA")
    le.fit(all_vals)
    encoders[col] = le

# ── 6. Build ranker datasets ───────────────────────────────────────────────────
def prepare_ranker_df(df, cands):
    merged = cands.merge(
        df[['user_id', 'video_id', 'label'] + feature_cols],
        on=['user_id', 'video_id'], how='inner'
    )
    merged['user_idx']  = merged['user_id'].map(user2idx).fillna(-1).astype(int)
    merged['video_idx'] = merged['video_id'].map(video2idx).fillna(-1).astype(int)
    merged = merged[(merged['user_idx'] >= 0) & (merged['video_idx'] >= 0)]
    for col in cat_cols:
        merged[col] = encoders[col].transform(merged[col].astype(str).fillna("NA"))
    merged[num_cols] = merged[num_cols].fillna(0.0)
    return merged

ranker_train = prepare_ranker_df(train_df, candidates)
ranker_test  = prepare_ranker_df(test_df,  candidates)

# Normalize numeric features using train stats
num_means = ranker_train[num_cols].mean()
num_stds  = ranker_train[num_cols].std().replace(0, 1)
ranker_train[num_cols] = (ranker_train[num_cols] - num_means) / num_stds
ranker_test[num_cols]  = (ranker_test[num_cols]  - num_means) / num_stds

cat_cardinalities = {col: int(ranker_train[col].max()) + 2 for col in cat_cols}


class RankerDataset(Dataset):
    def __init__(self, df):
        self.user_idx  = torch.LongTensor(df['user_idx'].values)
        self.video_idx = torch.LongTensor(df['video_idx'].values)
        self.num_feats = torch.FloatTensor(df[num_cols].values)
        self.cat_feats = torch.LongTensor(df[cat_cols].values)
        self.mf_score  = torch.FloatTensor(df['mf_score'].values)
        self.labels    = torch.FloatTensor(df['label'].values)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.user_idx[idx], self.video_idx[idx],
            self.num_feats[idx], self.cat_feats[idx],
            self.mf_score[idx],  self.labels[idx],
        )