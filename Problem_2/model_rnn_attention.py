import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from generate_utils import sample_next

# --- 1. OLD ARCHITECTURE (Seq2Seq / Bahdanau) ---
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

class RNNAttention_Old(nn.Module):
    """The Seq2Seq Encoder-Decoder model (will look ahead during training)"""
    def __init__(self, vocab_size, embed_dim=32, hidden_size=128, num_layers=1, attention_dim=64, dropout=0.3):
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

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @torch.no_grad()
    def generate(self, sos_idx, eos_idx, max_length=20, temperature=0.8, device=torch.device('cpu'), min_len=3):
        self.eval()
        
        # Start with the SOS token
        current_seq = torch.tensor([[sos_idx]], device=device)
        result = []
        
        for step in range(max_length):
            # Forward pass to get log probabilities
            log_probs, _ = self.forward(current_seq)
            
            # We only care about the prediction for the last character
            next_char_log_probs = log_probs[0, -1, :] 
            
            # Sample the next character
            idx = sample_next(next_char_log_probs, temperature=temperature, top_k=8,
                              repetition_penalty=1.5, generated_so_far=result,
                              eos_idx=eos_idx, min_length_reached=(step >= min_len))
            
            if idx == eos_idx: break
            result.append(idx)
            
            # Append the new character to our sequence and repeat
            next_char_tensor = torch.tensor([[idx]], device=device)
            current_seq = torch.cat([current_seq, next_char_tensor], dim=1)
            
        return result
    
    

# --- 2. NEW ARCHITECTURE (Autoregressive / Causal) ---
class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        Q, K, V = self.query(x), self.key(x), self.value(x)
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(x.size(-1))
        causal_mask = torch.tril(torch.ones(x.size(1), x.size(1), device=x.device)).unsqueeze(0)
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        return torch.bmm(F.softmax(scores, dim=-1), V)

class RNNAttention_New(nn.Module):
    """The corrected Autoregressive Causal Attention model"""
    def __init__(self, vocab_size, embed_dim=32, hidden_size=128, num_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_size, num_layers, batch_first=True)
        self.attention = CausalSelfAttention(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, hidden=None):
        emb = self.dropout(self.embedding(x))
        gru_out, hidden = self.gru(emb, hidden)
        ctx = self.attention(gru_out)
        combined = torch.cat([gru_out, ctx], dim=-1)
        out = self.fc(self.dropout(combined))
        return self.log_softmax(out), hidden

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @torch.no_grad()
    def generate(self, sos_idx, eos_idx, max_length=20, temperature=0.8, device=torch.device('cpu'), min_len=3):
        self.eval()
        
        # Start with the SOS token
        current_seq = torch.tensor([[sos_idx]], device=device)
        result = []
        
        for step in range(max_length):
            # Forward pass to get log probabilities
            log_probs, _ = self.forward(current_seq)
            
            # We only care about the prediction for the last character
            next_char_log_probs = log_probs[0, -1, :] 
            
            # Sample the next character
            idx = sample_next(next_char_log_probs, temperature=temperature, top_k=8,
                              repetition_penalty=1.5, generated_so_far=result,
                              eos_idx=eos_idx, min_length_reached=(step >= min_len))
            
            if idx == eos_idx: break
            result.append(idx)
            
            # Append the new character to our sequence and repeat
            next_char_tensor = torch.tensor([[idx]], device=device)
            current_seq = torch.cat([current_seq, next_char_tensor], dim=1)
            
        return result