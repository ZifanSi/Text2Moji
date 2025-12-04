import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim*2, 1)

    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden*2)
        weights = torch.tanh(self.attn(lstm_output))  # (batch, seq_len, 1)
        weights = torch.softmax(weights, dim=1)      # softmax over seq_len
        weighted = lstm_output * weights             # element-wise multiply
        representation = weighted.sum(dim=1)         # sum over seq_len
        return representation

class EnhancedBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, embedding_matrix=None,
                 freeze_embeddings=True, num_layers=2, dropout=0.3):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
            self.embedding.weight.requires_grad = not freeze_embeddings

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers>1 else 0
        )

        # Attention pooling
        self.attention = Attention(hidden_dim)

        # Classifier
        self.fc = nn.Linear(hidden_dim*2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)
        emb = self.dropout(emb)

        lstm_out, (h, c) = self.lstm(emb) 

        # Attention pooling
        rep = self.attention(lstm_out)
        rep = self.dropout(rep)

        logits = self.fc(rep)
        return logits
