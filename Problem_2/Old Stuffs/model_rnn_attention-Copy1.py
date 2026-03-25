import torch
import torch.nn as nn
import torch.nn.functional as F
from generate_utils import sample_next

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size, attention_dim):
        super().__init__()
        self.W = nn.Linear(hidden_size, attention_dim, bias=False)
        self.U = nn.Linear(hidden_size, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, enc_out, dec_hidden):
        energy = self.v(torch.tanh(self.W(enc_out) + self.U(dec_hidden).unsqueeze(1))).squeeze(-1)
        attn   = F.softmax(energy, dim=-1)
        return torch.bmm(attn.unsqueeze(1), enc_out).squeeze(1), attn

class RNNWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim=8, hidden_size=24,
                 num_layers=1, attention_dim=12, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder   = nn.GRU(embed_dim, hidden_size, num_layers, batch_first=True)
        self.attention = BahdanauAttention(hidden_size, attention_dim)
        self.decoder   = nn.GRU(embed_dim + hidden_size, hidden_size, num_layers, batch_first=True)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_size, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def encode(self, src):
        return self.encoder(self.dropout(self.embedding(src)))

    def decode_step(self, tgt_char, dec_hidden, enc_out):
        emb = self.dropout(self.embedding(tgt_char))
        ctx, attn = self.attention(enc_out, dec_hidden[-1])
        out, dec_hidden = self.decoder(torch.cat([emb, ctx], -1).unsqueeze(1), dec_hidden)
        return self.log_softmax(self.fc(self.dropout(out.squeeze(1)))), dec_hidden, attn

    def forward(self, x, hidden=None):
        enc_out, dec_hidden = self.encode(x)
        outputs = []
        for t in range(x.size(1) - 1):
            lp, dec_hidden, _ = self.decode_step(x[:, t], dec_hidden, enc_out)
            outputs.append(lp)
        return torch.stack(outputs, dim=1), dec_hidden

    @torch.no_grad()
    def generate(self, sos_idx, eos_idx, max_length=20,
                 temperature=0.8, device=torch.device('cpu'), min_len=3):
        self.eval()
        src = torch.tensor([[sos_idx]], device=device)
        enc_out, dec_hidden = self.encode(src)
        tgt, result = torch.tensor([sos_idx], device=device), []
        for step in range(max_length):
            lp, dec_hidden, _ = self.decode_step(tgt, dec_hidden, enc_out)
            idx = sample_next(lp[0], temperature=temperature, top_k=8,
                              repetition_penalty=1.5, generated_so_far=result,
                              eos_idx=eos_idx, min_length_reached=(step >= min_len))
            if idx == eos_idx: break
            result.append(idx)
            tgt = torch.tensor([idx], device=device)
        return result

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
