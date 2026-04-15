"""
Feature Engineering for KuaiRand-Pure
Based on: "Causal Discovery in Recommender Systems" (CONSEQUENCES 2024)
          "KuaiRand: An Unbiased Sequential Recommendation Dataset" (CIKM 2022)

Causal tier structure (from paper):
    # A user with certain characteristics (tier 1) appears in a context (tier 2)
      saw a video with certain properties (tier 3) , this video has acummulated engagement
      (tier 4) and, because of this, the user response of certain way (tier 5)

    Tier 1 - User features       → user_active_degree, follow/fans counts, register_days
    Tier 2 - Context feature     → tab (policy/context)
    Tier 3 - Item features       → video_type, video_duration, tag
    Tier 4 - Item statistics     → completion_rate, like_rate, share_rate, virality
    Tier 5 - Feedback signals    → watch_ratio (label)

Excluded (per causal paper):
    - onehot_feat* : encrypted, no semantic meaning for causal modeling
    - music_id     : near-unique per video, no generalizable signal
"""

import pandas as pd
import numpy as np
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'kuairand', 'KuaiRand-Pure', 'data'
)
OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'training_data'
)
os.makedirs(OUT_DIR, exist_ok=True)

def src(filename):
    return os.path.join(BASE_DIR, filename)

def out(filename):
    return os.path.join(OUT_DIR, filename)

print("=" * 60)
print("KuaiRand Feature Engineering Pipeline")
print("=" * 60)

# ── 1. Load raw data ───────────────────────────────────────────────────────────
print("\n[1/6] Loading raw data...")

df_rand  = pd.read_csv(src("log_random_4_22_to_5_08_pure.csv"))
df_std1  = pd.read_csv(src("log_standard_4_08_to_4_21_pure.csv"))
df_std2  = pd.read_csv(src("log_standard_4_22_to_5_08_pure.csv"))
users    = pd.read_csv(src("user_features_pure.csv"))
vid_basic = pd.read_csv(src("video_features_basic_pure.csv"))
vid_stats = pd.read_csv(src("video_features_statistic_pure.csv"))

print(f"  log_random rows      : {len(df_rand):,}")
print(f"  log_standard rows    : {len(df_std1) + len(df_std2):,}")
print(f"  users                : {len(users):,}")
print(f"  videos (basic)       : {len(vid_basic):,}")
print(f"  videos (statistics)  : {len(vid_stats):,}")

# ── 2. Build interaction log ───────────────────────────────────────────────────
print("\n[2/6] Building interaction log...")

# Tag the source so we can split later (random = unbiased ground truth)
df_rand['is_random'] = 1
df_std1['is_random'] = 0
df_std2['is_random'] = 0

interactions = pd.concat([df_rand, df_std1, df_std2], ignore_index=True)


# Keep only the columns present in all three files
keep_cols = [c for c in interactions.columns if c != 'tab' or 'tab' in interactions.columns]
interactions = interactions[keep_cols]



print(f"  Total interactions   : {len(interactions):,}")
print(f"  Columns              : {interactions.columns.tolist()}")




# ── 3. Tier 1 — User features (causal paper) ──────────────────────────────────
print("\n[3/6] Building Tier-1 user features...")

# Drop encrypted onehot features (no semantic meaning per causal paper)
onehot_cols = [c for c in users.columns if c.startswith('onehot_feat')]
users_clean = users.drop(columns=onehot_cols)

# Derived: social influence ratio (fans vs follow)
users_clean['fans_follow_ratio'] = (
    users_clean['fans_user_num'] / (users_clean['follow_user_num'] + 1)
).round(4)

# Derived: creator flag (is_video_author OR is_live_streamer) only 1 field
users_clean['is_creator'] = (
    (users_clean['is_video_author'] == 1) | (users_clean['is_live_streamer'] == 1)
).astype(int)

# Keep only causally relevant columns (Tier 1)
user_tier1_cols = [
    'user_id',
    'user_active_degree',
    'is_lowactive_period',
    'is_creator',
    'follow_user_num',
    'follow_user_num_range',
    'fans_user_num',
    'fans_user_num_range',
    'fans_follow_ratio',
    'friend_user_num',
    'friend_user_num_range',
    'register_days',
    'register_days_range',
]
users_final = users_clean[user_tier1_cols]
print(f"  User features shape  : {users_final.shape}")




# ── 4. Tier 3+4 — Video features (basic + statistics) ─────────────────────────
print("\n[4/6] Building Tier-3/4 video features...")

# Merge basic + statistics on video_id
# Drop music_id (near-unique, excluded by causal paper)
vid_basic_clean = vid_basic.drop(columns=['music_id'], errors='ignore')

#Merge beewteen videos and videos statistics
videos = vid_basic_clean.merge(vid_stats, on='video_id', how='inner')



# ── Tier 3: Item metadata features
# Convert upload_dt to days-since-upload (recency)
videos['upload_dt'] = pd.to_datetime(videos['upload_dt'], errors='coerce')
reference_date = videos['upload_dt'].max()
videos['days_since_upload'] = (reference_date - videos['upload_dt']).dt.days.fillna(-1).astype(int)

# Aspect ratio
videos['aspect_ratio'] = (
    videos['server_width'] / (videos['server_height'] + 1)
).round(3)


# ── Tier 4: Engagement rate features (key insight from causal paper)
eps = 1e-6  # avoid division by zero

# Completion rate: most important signal (per KuaiRand paper)
videos['completion_rate'] = (
    videos['complete_play_cnt'] / (videos['play_cnt'] + eps)
).clip(0, 1).round(4)

# Like rate
videos['like_rate'] = (
    videos['like_cnt'] / (videos['play_cnt'] + eps)
).clip(0, 1).round(4)

# Share rate (virality)
videos['share_rate'] = (
    videos['share_cnt'] / (videos['play_cnt'] + eps)
).clip(0, 1).round(4)

# Follow rate (strongest intent signal)
videos['follow_rate'] = (
    videos['follow_cnt'] / (videos['play_cnt'] + eps)
).clip(0, 1).round(4)

# Comment rate
videos['comment_rate'] = (
    videos['comment_cnt'] / (videos['play_cnt'] + eps)
).clip(0, 1).round(4)

# Negative signal: report rate
videos['report_rate'] = (
    videos['report_cnt'] / (videos['play_cnt'] + eps)
).clip(0, 1).round(4)

# Composite engagement score (weighted per signal importance in literature)
videos['engagement_score'] = (
    0.40 * videos['completion_rate'] +
    0.25 * videos['like_rate'] +
    0.20 * videos['share_rate'] +
    0.10 * videos['follow_rate'] +
    0.05 * videos['comment_rate']
).round(4)

# Average view duration per play
videos['avg_play_duration'] = (
    videos['play_duration'] / (videos['play_cnt'] + eps)
).round(2)

# Keep only useful video columns
video_final_cols = [
    # Tier 3 stats
    'video_id',
    'author_id',
    'video_type',
    'upload_type',
    'video_duration',
    'days_since_upload',
    'aspect_ratio',
    'music_type',
    'tag',
    # Tier 4 stats
    'show_cnt',
    'play_cnt',
    'completion_rate',
    'like_rate',
    'share_rate',
    'follow_rate',
    'comment_rate',
    'report_rate',
    'avg_play_duration',
    'engagement_score',
]
videos_final = videos[video_final_cols]
print(f"  Video features shape : {videos_final.shape}")



# ── 5. Build training dataset ──────────────────────────────────────────────────
print("\n[5/6] Joining interactions with user and video features...")

# Join all tiers
df = interactions.merge(users_final,  on='user_id',  how='left')
df = df.merge(videos_final, on='video_id', how='left')

# ── Label: PCR-based formula (KDD 2024 "Counteracting Duration Bias" paper)
df['pcr'] = (df['play_time_ms'] / df['duration_ms'].replace(0, float('nan'))).clip(0, 1)

w_07 = df['duration_ms'].quantile(0.70)
print(f"  Duration p70 threshold: {w_07/1000:.1f}s")

short_and_complete = (df['duration_ms'] <= w_07) & (df['pcr'] >= 0.99)
long_and_enough    = (df['duration_ms'] >  w_07) & (df['pcr'] >= 0.70)
df['label_watch']  = (short_and_complete | long_and_enough).astype(int)

df['label_explicit'] = (
    (df['is_like']    == 1) |
    (df['is_follow']  == 1) |
    (df['is_forward'] == 1)
).astype(int)

df['label'] = ((df['label_watch'] == 1) | (df['label_explicit'] == 1)).astype(int)

before = len(df)
df = df.dropna(subset=['user_active_degree', 'completion_rate'])
after  = len(df)

print(f"  Rows after join       : {before:,} → {after:,} (dropped {before-after:,} unmatched)")
print(f"  label_watch=1         : {df['label_watch'].sum():,}  ({df['label_watch'].mean()*100:.1f}%)")
print(f"  label_explicit=1      : {df['label_explicit'].sum():,}  ({df['label_explicit'].mean()*100:.1f}%)")
print(f"  label final=1         : {df['label'].sum():,}  ({df['label'].mean()*100:.1f}%)")

# Label distribution by split
print(f"\n  Label distribution by split:")
for split_val, split_name in [(0, 'train (standard)'), (1, 'test (random)')]:
    subset = df[df['is_rand'] == split_val]
    pos = subset['label'].mean() * 100
    print(f"    {split_name}: {len(subset):,} rows  |  positives: {pos:.1f}%")

print(f"\n  Final shape           : {df.shape}")

df = df.drop(columns=['pcr', 'label_watch', 'label_explicit'])


# ── 6. Save outputs ────────────────────────────────────────────────────────────
print("\n[6/6] Saving to training_data/ ...")

# Split using is_rand column (1 = randomly exposed = test, 0 = standard = train)
train_df = df[df['is_rand'] == 0].drop(columns=['is_rand'])
test_df  = df[df['is_rand'] == 1].drop(columns=['is_rand'])

train_df.to_csv(out("train.csv"), index=False)
test_df.to_csv(out("test.csv"),   index=False)
users_final.to_csv(out("user_features.csv"),  index=False)
videos_final.to_csv(out("video_features.csv"), index=False)

print(f"  train.csv            : {len(train_df):,} rows  → training_data/train.csv")
print(f"  test.csv             : {len(test_df):,} rows   → training_data/test.csv")
print(f"  user_features.csv    : {len(users_final):,} users  → training_data/user_features.csv")
print(f"  video_features.csv   : {len(videos_final):,} videos → training_data/video_features.csv")

print("\n" + "=" * 60)
print("Done. Feature summary:")
print("=" * 60)

print(f"\nTier 1 - User features ({len(user_tier1_cols)-1} features):")
for c in user_tier1_cols[1:]:
    print(f"  {c}")

print(f"\nTier 3+4 - Video features ({len(video_final_cols)-1} features):")
for c in video_final_cols[1:]:
    print(f"  {c}")

print(f"\nLabel: PCR + explicit signals → binary 'label' (1 = engaged, 0 = not engaged)")
print(f"Train/test split: is_rand==0 → train | is_rand==1 → test (unbiased)")