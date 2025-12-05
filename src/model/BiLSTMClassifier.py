import torch
import torch.nn as nn
import torch.nn.functional as F

class OptimizedAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.context_vector = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_dim * 2)
        
        # u = tanh(W * h + b)
        u = torch.tanh(self.projection(lstm_output))  # (batch, seq_len, hidden_dim)
        
        # scores = u^T * u_w
        scores = self.context_vector(u)               # (batch, seq_len, 1)
        
        # Softmax
        weights = F.softmax(scores, dim=1)            # (batch, seq_len, 1)

        weighted_output = lstm_output * weights       # (batch, seq_len, hidden*2)
        representation = weighted_output.sum(dim=1)   # (batch, hidden*2)
        
        return representation

class EnhancedBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, embedding_matrix=None,
                 freeze_embeddings=True, num_layers=2, dropout=0.5):
        super().__init__()

        # 1. Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
            self.embedding.weight.requires_grad = not freeze_embeddings

        # 2. LSTM Layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 3. Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

        # 4. Optimized Attention
        self.attention = OptimizedAttention(hidden_dim)

        # 5. Classifier
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # Embedding
        emb = self.embedding(x)
        emb = self.dropout(emb)

        # LSTM
        # output: (batch, seq_len, hidden*2)
        lstm_out, _ = self.lstm(emb)
        
        # Apply LayerNorm before Attention
        lstm_out = self.layer_norm(lstm_out)

        # Attention Pooling
        rep = self.attention(lstm_out)
        
        # Final Classification
        rep = self.dropout(rep)
        logits = self.fc(rep)
        
        return logits