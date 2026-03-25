import torch
from data_utils import load_names,Vocabulary
from model_blstm import BLSTM
from model_rnn_attention import RNNWithAttention

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab=torch.load("checkpoints/vocab.pt",map_location="cpu",weights_only=False)
sos,eos=vocab.char2idx["<SOS>"],vocab.char2idx["<EOS>"]

print(f"SOS={sos} EOS={eos} vocab_size={len(vocab)}\n")


print("=== BLSTM:manual step through ===")
model=BLSTM(len(vocab),embed_dim=8,hidden_size=24,num_layers=2,dropout=0.0)
model.load_state_dict(torch.load("checkpoints/blstm.pt",map_location=device,weights_only=True))
model.to(device).eval()

with torch.no_grad() :
    x,hidden=torch.tensor([[sos]],device=device), None
    for step in range(12) :
        lp,hidden=model(x,hidden)
        probs=torch.exp(lp[0, 0])
        top5_p,top5_i=torch.topk(probs,5)
        top5=[(vocab.idx2char.get(i.item(),'?'),f"{p.item():.3f}")for i,p in zip(top5_i,top5_p)]
        last_char=vocab.idx2char.get(x[0,0].item(), '?')
        print(f"  step {step:2d} input='{last_char}' top5={top5}")
        # pick argmax (greedy) to see where it wants to go
        best=top5_i[0].item()
        if best == eos :
            print("-> EOS predicted,stopping")
            break
        x=torch.tensor([[best]],device=device)

print()

# ── Test with repetition penalty manually ────────────────────────────────────
print("=== BLSTM: with repetition_penalty=3.0 ===")
model2=BLSTM(len(vocab),embed_dim=8,hidden_size=24,num_layers=2,dropout=0.0)
model2.load_state_dict(torch.load("checkpoints/blstm.pt",map_location=device,weights_only=True))
model2.to(device).eval()

from generate_utils import sample_next
result,x,hidden=[],torch.tensor([[sos]],device=device), None
with torch.no_grad():
    for step in range(20):
        lp, hidden=model2(x,hidden)
        idx=sample_next(lp[0,0],temperature=0.8,top_k=8,
                         repetition_penalty=3.0,generated_so_far=result,
                         eos_idx=eos,min_length_reached=(step>=3))
        if idx==eos : break
        result.append(idx)
        x=torch.tensor([[idx]],device=device)

name=vocab.decode(result)
print(f"penalty=3.0 ->'{name}'")

# Try penalty=5.0
result2,x2,hidden2=[],torch.tensor([[sos]],device=device), None
with torch.no_grad() :
    for step in range(20) :
        lp, hidden2=model2(x2,hidden2)
        idx=sample_next(lp[0,0],temperature=0.8,top_k=8,
                         repetition_penalty=5.0,generated_so_far=result2,
                         eos_idx=eos,min_length_reached=(step>=3))
        if idx==eos : break
        result2.append(idx)
        x2=torch.tensor([[idx]],device=device)

name2=vocab.decode(result2)
print(f"penalty=5.0 ->'{name2}'")
