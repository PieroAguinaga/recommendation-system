"""model.py"""
import torch
import torch.nn as nn
from .dataset import (
    N_USERS, N_VIDEOS, model,
    cat_cols, EMBEDDING_DIM,
    cat_cardinalities,  
)

class NeuralRanker(nn.Module):
    """
    Concatena: user_emb(64) + video_emb(64) + num_feats + cat_embs + mf_score
    → MLP [256→128→64→1] → sigmoid → P(engagement)
    Los embeddings del MF se cargan congelados.
    """
    def __init__(self, n_num, cat_cardinalities, emb_dim, hidden_dims, dropout):
        super().__init__()
        # Pretrained MF embeddings — frozen
        self.user_emb_layer  = nn.Embedding(N_USERS,  emb_dim)
        self.video_emb_layer = nn.Embedding(N_VIDEOS, emb_dim)
        self.user_emb_layer.weight.data  = model.user_emb.weight.data.clone()
        self.video_emb_layer.weight.data = model.video_emb.weight.data.clone()
        self.user_emb_layer.weight.requires_grad  = False
        self.video_emb_layer.weight.requires_grad = False

        # Small learned embeddings for categoricals (4 dims each)
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card, 4) for card in cat_cardinalities.values()
        ])
        input_size = emb_dim + emb_dim + n_num + len(cat_cols) * 4 + 1
        layers = []
        prev = input_size
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_idx, video_idx, num_feats, cat_feats, mf_score):
        u = self.user_emb_layer(user_idx)
        v = self.video_emb_layer(video_idx)
        cat_embs = torch.cat(
            [self.cat_embeddings[i](cat_feats[:, i])
             for i in range(len(self.cat_embeddings))],
            dim=1
        )
        x = torch.cat([u, v, num_feats, cat_embs, mf_score.unsqueeze(1)], dim=1)
        return torch.sigmoid(self.mlp(x)).squeeze()