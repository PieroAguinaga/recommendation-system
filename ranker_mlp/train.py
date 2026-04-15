"""train.py"""
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from .dataset import (
    ranker_train, ranker_test,
    ranker_train_loader, ranker_val_loader,
    num_cols, cat_cols, cat_cardinalities,
    EMBEDDING_DIM, DEVICE, OUT_DIR,
)
from .model import NeuralRanker

# ── Hyperparameters ────────────────────────────────────────────────────────────
RANKER_EPOCHS = 20
RANKER_LR     = 1e-3
HIDDEN_DIMS   = [256, 128, 64]
DROPOUT       = 0.3
TOP_K_FINAL   = 10
best_ranker_path = OUT_DIR / "neural_ranker_best.pt"

# ── Build model ────────────────────────────────────────────────────────────────
ranker_model = NeuralRanker(
    n_num=len(num_cols),
    cat_cardinalities=cat_cardinalities,
    emb_dim=EMBEDDING_DIM,
    hidden_dims=HIDDEN_DIMS,
    dropout=DROPOUT,
).to(DEVICE)

n_params = sum(p.numel() for p in ranker_model.parameters() if p.requires_grad)
print(f'Trainable parameters: {n_params:,}')

ranker_optimizer = torch.optim.Adam(ranker_model.parameters(), lr=RANKER_LR, weight_decay=1e-5)
ranker_criterion = nn.BCELoss()
ranker_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(ranker_optimizer, patience=3, factor=0.5)


best_ranker_auc  = 0.0
best_ranker_path = 'training_data/neural_ranker_best.pt'

print(f'Training Neural Ranker on {DEVICE}...\n')

for epoch in range(1, RANKER_EPOCHS + 1):
    # Train
    ranker_model.train()
    total_loss = 0.0
    for u, v, num, cat, mf, y in ranker_train_loader:
        u, v, num, cat, mf, y = (
            u.to(DEVICE), v.to(DEVICE), num.to(DEVICE),
            cat.to(DEVICE), mf.to(DEVICE), y.to(DEVICE)
        )
        preds = ranker_model(u, v, num, cat, mf)
        loss  = ranker_criterion(preds, y)
        ranker_optimizer.zero_grad()
        loss.backward()
        ranker_optimizer.step()
        total_loss += loss.item() * len(y)
    avg_loss = total_loss / len(ranker_train)

    # Validate
    ranker_model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for u, v, num, cat, mf, y in ranker_val_loader:
            u, v, num, cat, mf = (
                u.to(DEVICE), v.to(DEVICE), num.to(DEVICE),
                cat.to(DEVICE), mf.to(DEVICE)
            )
            all_preds.extend(ranker_model(u, v, num, cat, mf).cpu().numpy())
            all_labels.extend(y.numpy())

    auc = roc_auc_score(all_labels, all_preds)
    ranker_scheduler.step(1 - auc)
    marker = ' ← best' if auc > best_ranker_auc else ''
    print(f'  Epoch {epoch:02d}/{RANKER_EPOCHS}  loss: {avg_loss:.4f}  AUC: {auc:.4f}{marker}')
    if auc > best_ranker_auc:
        best_ranker_auc = auc
        torch.save(ranker_model.state_dict(), best_ranker_path)

print(f'\nBest AUC: {best_ranker_auc:.4f}')
print(f'Model saved → {best_ranker_path}')





# ── Final evaluation & feed generation ────────────────────────────────────────
ranker_model.load_state_dict(torch.load(best_ranker_path, map_location=DEVICE))
ranker_model.eval()

final_preds = []
with torch.no_grad():
    for u, v, num, cat, mf, y in ranker_val_loader:
        u, v, num, cat, mf = (
            u.to(DEVICE), v.to(DEVICE), num.to(DEVICE),
            cat.to(DEVICE), mf.to(DEVICE)
        )
        final_preds.extend(ranker_model(u, v, num, cat, mf).cpu().numpy())

ranker_test = ranker_test.copy()
ranker_test['neural_score'] = final_preds

# AUC comparison
auc_neural = roc_auc_score(ranker_test['label'], ranker_test['neural_score'])
auc_mf     = roc_auc_score(ranker_test['label'], ranker_test['mf_score'])
print(f'MF score AUC    : {auc_mf:.4f}  (baseline)')
print(f'Neural ranker AUC: {auc_neural:.4f}')
print(f'Improvement      : +{(auc_neural - auc_mf)*100:.2f} AUC points\n')

# Precision & Recall @K
for K in [10, 50, 100]:
    results_k = []
    for uid, grp in ranker_test.groupby('user_id'):
        if grp['label'].sum() == 0:
            continue
        top_k = grp.nlargest(K, 'neural_score')
        hits  = top_k['label'].sum()
        results_k.append({'precision': hits / K, 'recall': hits / grp['label'].sum()})
    results_k = pd.DataFrame(results_k)
    print(f'  Precision@{K}: {results_k["precision"].mean():.4f}  '
          f'|  Recall@{K}: {results_k["recall"].mean():.4f}')

# Generate top-10 feed per user
final_feed = (
    ranker_test
    .sort_values('neural_score', ascending=False)
    .groupby('user_id')
    .head(TOP_K_FINAL)
    .sort_values(['user_id', 'neural_score'], ascending=[True, False])
    [['user_id', 'video_id', 'neural_score', 'mf_score', 'label']]
    .reset_index(drop=True)
)
final_feed.to_csv('training_data/ranked_candidates_neural.csv', index=False)
print(f'\nFeed saved → training_data/ranked_candidates_neural.csv  ({len(final_feed):,} rows)')
final_feed[final_feed['user_id'] == final_feed['user_id'].iloc[0]]