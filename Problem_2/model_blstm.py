import torch
import torch.nn as nn
from generate_utils import sample_next

class BLSTM_Old(nn.Module) :
    """The original bidirectional model (will overfit/fail generation)"""
    def __init__(self,vocab_size,embed_dim=32,hidden_size=128,num_layers=2,dropout=0.5) :
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,embed_dim,padding_idx=0)
        self.lstm=nn.LSTM(embed_dim,hidden_size,num_layers, 
                            bidirectional=True,batch_first=True,dropout=dropout)
  
        self.fc=nn.Linear(hidden_size*2,vocab_size)
        self.log_softmax=nn.LogSoftmax(dim=-1)

    def forward(self,x,hidden=None) :
        emb=self.embedding(x)
        out,hidden=self.lstm(emb,hidden)
        return self.log_softmax(self.fc(out)),hidden

    def count_parameters(self) :
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @torch.no_grad()
    def generate(self,sos_idx,eos_idx,max_length=20,temperature=0.8,device=torch.device('cpu'), min_len=3) :
        self.eval()
        
        # Start with the SOS token
        current_seq=torch.tensor([[sos_idx]],device=device)
        result=[]
        
        for step in range(max_length) :
            # Forward pass to get log probabilities
            log_probs,_=self.forward(current_seq)
            
            # We only care about the prediction for the last character
            next_char_log_probs=log_probs[0,-1,:] 
            
            # Sample the next character
            idx=sample_next(next_char_log_probs,temperature=temperature,top_k=8,
                              repetition_penalty=1.5,generated_so_far=result,
                              eos_idx=eos_idx,min_length_reached=(step>=min_len))
            
            if idx==eos_idx: break
            result.append(idx)
            
            # Append the new character to our sequence and repeat
            next_char_tensor=torch.tensor([[idx]],device=device)
            current_seq=torch.cat([current_seq,next_char_tensor],dim=1)
            
        return result
    
    

class BLSTM_New(nn.Module) :
    
    def __init__(self,vocab_size,embed_dim=32,hidden_size=128,num_layers=2,dropout=0.5) :
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,embed_dim,padding_idx=0)
        self.lstm=nn.LSTM(embed_dim,hidden_size,num_layers, 
                            bidirectional=False,batch_first=True,dropout=dropout)
        self.fc=nn.Linear(hidden_size,vocab_size)
        self.log_softmax=nn.LogSoftmax(dim=-1)

    def forward(self,x,hidden=None) :
        emb=self.embedding(x)
        out,hidden=self.lstm(emb,hidden)
        return self.log_softmax(self.fc(out)),hidden
        
    def count_parameters(self) :
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @torch.no_grad()
    def generate(self,sos_idx,eos_idx,max_length=20,temperature=0.8,device=torch.device('cpu'), min_len=3) :
        self.eval()
        
        # Start with the SOS token
        current_seq=torch.tensor([[sos_idx]],device=device)
        result=[]
        
        for step in range(max_length) :
            # Forward pass to get log probabilities
            log_probs,_=self.forward(current_seq)
            
            # We only care about the prediction for the last character
            next_char_log_probs=log_probs[0,-1,:] 
            
            # Sample the next character
            idx=sample_next(next_char_log_probs,temperature=temperature,top_k=8,
                              repetition_penalty=1.5,generated_so_far=result,
                              eos_idx=eos_idx,min_length_reached=(step>=min_len))
            
            if idx==eos_idx: break
            result.append(idx)
            
            # Append the new character to our sequence and repeat
            next_char_tensor=torch.tensor([[idx]],device=device)
            current_seq=torch.cat([current_seq,next_char_tensor],dim=1)
            
        return result