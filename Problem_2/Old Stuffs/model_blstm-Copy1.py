import torch
import torch.nn as nn
from generate_utils import sample_next

class BLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=8, hidden_size=24,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding   = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm        = nn.LSTM(embed_dim, hidden_size, num_layers,
                                   batch_first=True, bidirectional=True,
                                   dropout=dropout if num_layers > 1 else 0)
        self.dropout     = nn.Dropout(dropout)
        self.proj        = nn.Linear(hidden_size * 2, hidden_size)
        self.fc          = nn.Linear(hidden_size, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, hidden=None):
        emb = self.dropout(self.embedding(x))
        out, hidden = self.lstm(emb, hidden)
        out = torch.relu(self.proj(self.dropout(out)))
        return self.log_softmax(self.fc(out)), hidden

    @torch.no_grad()
    def generate(self, sos_idx, eos_idx, max_length=20,
                 temperature=0.8, device=torch.device('cpu'), min_len=4): # Increased min_len
        self.eval()
        x, hidden, result = torch.tensor([[sos_idx]], device=device), None, []
        for step in range(max_length):
            lp, hidden = self(x, hidden)
            # Change repetition_penalty from 1.5 to 3.0
            idx = sample_next(lp[0, 0], temperature=temperature, top_k=8,
                              repetition_penalty=3.0, generated_so_far=result,
                              eos_idx=eos_idx, min_length_reached=(step >= min_len))
            if idx == eos_idx: break
            result.append(idx)
            x = torch.tensor([[idx]], device=device)
        return result

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
