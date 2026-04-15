import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os
from dataset import InteractionDataset, loader, N_USERS, N_VIDEOS
from model import MatrixFactorization

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path(os.path.dirname(os.path.abspath(__file__))).parent
TRAIN_PATH = BASE / "training_data" / "train.csv"
OUT_DIR    = BASE / "training_data"
 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
 
# ── Hyperparameters ────────────────────────────────────────────────────────────
EMBEDDING_DIM = 64      # size of each user/video vector
EPOCHS        = 10
LR            = 1e-3
WEIGHT_DECAY  = 1e-5
TOP_K         = 500     # candidates to retrieve per user
 

print("\n Training Matrix Factorization model...")
 
model     = MatrixFactorization(N_USERS, N_VIDEOS, EMBEDDING_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.BCELoss()
 
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for users, videos, labels in loader:
        users, videos, labels = users.to(DEVICE), videos.to(DEVICE), labels.to(DEVICE)
 
        preds = model(users, videos)
        loss  = criterion(preds, labels)
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        total_loss += loss.item() * len(labels)
 
    avg_loss = total_loss / len(train)
    print(f"  Epoch {epoch:02d}/{EPOCHS}  |  loss: {avg_loss:.4f}")
 
# Save model
torch.save({
    "model_state": model.state_dict(),
    "user2idx":    user2idx,
    "video2idx":   video2idx,
    "idx2video":   {i: v for v, i in video2idx.items()},
    "config":      {"dim": EMBEDDING_DIM, "n_users": N_USERS, "n_videos": N_VIDEOS}
}, OUT_DIR / "mf_model.pt")
print(f"\n  Model saved → training_data/mf_model.pt")




print(f"\n Generating top-{TOP_K} candidates per user...")
 
model.eval()
results = []
 
# Get all video embeddings once (7583 × 64)
with torch.no_grad():
    all_video_emb  = model.get_all_video_embeddings()      # (N_VIDEOS, dim)
    all_video_bias = model.video_bias(
        torch.arange(N_VIDEOS).to(DEVICE)
    ).squeeze()                                             # (N_VIDEOS,)
 
BATCH_USERS = 256  # process users in batches to avoid OOM
user_idx_list = list(range(N_USERS))
 
for start in range(0, N_USERS, BATCH_USERS):
    batch_user_idx = torch.LongTensor(
        user_idx_list[start: start + BATCH_USERS]
    ).to(DEVICE)
 
    with torch.no_grad():
        u_emb  = model.user_emb(batch_user_idx)          # (B, dim)
        u_bias = model.user_bias(batch_user_idx).squeeze()  # (B,)
 
        # scores = u_emb @ all_video_emb.T + u_bias + video_bias
        scores = torch.sigmoid(
            u_emb @ all_video_emb.T                       # (B, N_VIDEOS)
            + u_bias.unsqueeze(1)
            + all_video_bias.unsqueeze(0)
        )                                                  # (B, N_VIDEOS)
 
        # top-K video indices per user
        topk_vals, topk_idx = torch.topk(scores, k=TOP_K, dim=1)
 
    topk_idx  = topk_idx.cpu().numpy()
    topk_vals = topk_vals.cpu().numpy()
 
    for i, user_idx_val in enumerate(batch_user_idx.cpu().numpy()):
        original_user_id = user_ids[user_idx_val]
        for rank, (vid_idx, score) in enumerate(zip(topk_idx[i], topk_vals[i])):
            results.append({
                "user_id":  original_user_id,
                "video_id": video_ids[vid_idx],
                "mf_score": round(float(score), 4),
                "mf_rank":  rank + 1, #Relative order (ranking)
            })
 
    if (start // BATCH_USERS) % 10 == 0:
        print(f"  Processed {min(start + BATCH_USERS, N_USERS):,} / {N_USERS:,} users")
 
candidates = pd.DataFrame(results)
candidates.to_csv(OUT_DIR / "candidates.csv", index=False)
 
print(f"\nCandidates saved → training_data/candidates.csv")
print(f"  Shape: {candidates.shape}")
print(f"  Sample:")
print(candidates.head(10).to_string(index=False))
 
print(f"\nDone.")
print(f"  Next step: join candidates.csv with train.csv features")
print(f"  and train a LightGBM ranker on top.")
 


from sklearn.metrics import roc_auc_score

print('Evaluating on test set (unbiased log_random)...')

test_data = test_df[['user_id', 'video_id', 'label']].copy()

# Only keep users and videos seen during training
test_data = test_data[
    test_data['user_id'].isin(user2idx) &
    test_data['video_id'].isin(video2idx)
].copy()

test_data['user_idx']  = test_data['user_id'].map(user2idx)
test_data['video_idx'] = test_data['video_id'].map(video2idx)

print(f'Test interactions: {len(test_data):,}')

# Predict scores in batches
model.eval()
all_preds = []
EVAL_BATCH = 8192

for start in range(0, len(test_data), EVAL_BATCH):
    batch = test_data.iloc[start:start+EVAL_BATCH]
    u = torch.LongTensor(batch['user_idx'].values).to(DEVICE)
    v = torch.LongTensor(batch['video_idx'].values).to(DEVICE)
    with torch.no_grad():
        preds = model(u, v).cpu().numpy()
    all_preds.extend(preds)

test_data['pred_score'] = all_preds

# AUC
auc = roc_auc_score(test_data['label'], test_data['pred_score'])
print(f'\nAUC: {auc:.4f}')

# Precision & Recall @K
for K in [10, 50, 100]:
    # Per user: top-K predicted vs actual positives
    results_k = []
    for uid, grp in test_data.groupby('user_id'):
        if grp['label'].sum() == 0:
            continue
        top_k = grp.nlargest(K, 'pred_score')
        hits  = top_k['label'].sum()
        prec  = hits / K
        rec   = hits / grp['label'].sum()
        results_k.append({'precision': prec, 'recall': rec})
    results_k = pd.DataFrame(results_k)
    print(f'  Precision@{K}: {results_k["precision"].mean():.4f}  |  Recall@{K}: {results_k["recall"].mean():.4f}')