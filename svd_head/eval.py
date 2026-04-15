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