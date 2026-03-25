import argparse, os, random, time
import torch, torch.nn as nn
from torch.utils.data import DataLoader

# --- LOCAL IMPORTS ---
from data_utils          import load_names, Vocabulary, NamesDataset, collate_fn
from model_vanilla_rnn   import VanillaRNN
from model_blstm         import BLSTM_Old, BLSTM_New
from model_rnn_attention import RNNAttention_Old, RNNAttention_New

DEFAULTS = dict(
    data_path="TrainingNames.txt", epochs=150, batch_size=16,
    val_split=0.15, model="all", ckpt_dir="checkpoints", seed=42, patience=20,
)

# Hyperparameters for all 5 models
HPARAMS = {
    "vanilla_rnn":       dict(embed_dim=32, hidden_size=128, num_layers=2, dropout=0.3, lr=1e-3, weight_decay=1e-4),
    "blstm_old":         dict(embed_dim=32, hidden_size=128, num_layers=2, dropout=0.5, lr=5e-4, weight_decay=1e-3),
    "blstm_new":         dict(embed_dim=32, hidden_size=128, num_layers=2, dropout=0.5, lr=5e-4, weight_decay=1e-3),
    "rnn_attention_old": dict(embed_dim=32, hidden_size=128, num_layers=1, attention_dim=64, dropout=0.3, lr=8e-4, weight_decay=1e-4),
    "rnn_attention_new": dict(embed_dim=32, hidden_size=128, num_layers=1, dropout=0.3, lr=8e-4, weight_decay=1e-4),
}

NON_MODEL = {'lr', 'weight_decay'}

def build_model(name, vocab_size):
    """Instantiates the requested model architecture."""
    hp = {k: v for k, v in HPARAMS[name].items() if k not in NON_MODEL}
    if name == "vanilla_rnn":         return VanillaRNN(vocab_size, **hp)
    elif name == "blstm_old":         return BLSTM_Old(vocab_size, **hp)
    elif name == "blstm_new":         return BLSTM_New(vocab_size, **hp)
    elif name == "rnn_attention_old": return RNNAttention_Old(vocab_size, **hp)
    elif name == "rnn_attention_new": return RNNAttention_New(vocab_size, **hp)
    else: raise ValueError(f"Unknown model: {name}")

def run_epoch(model, loader, optimizer, criterion, device, mname, train):
    model.train() if train else model.eval()
    total_loss, total_tok = 0.0, 0
    with (torch.enable_grad() if train else torch.no_grad()):
        for inp, tgt, lengths in loader:
            inp, tgt = inp.to(device), tgt.to(device)
            if train: optimizer.zero_grad()
            
            lp, _ = model(inp)
            
            # --- THE ARCHITECTURE HACK ---
            # The "Old" Seq2Seq models output a sequence that is 1 character shorter 
            # than the native autoregressive models. We must truncate the target to match.
            if lp.size(1) != tgt.size(1):
                tgt = tgt[:, :lp.size(1)]
            # -----------------------------
                
            B, T, V = lp.shape
            loss = criterion(lp.reshape(B*T, V), tgt.reshape(B*T))
            
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                optimizer.step()
                
            n = (tgt != 0).sum().item()
            total_loss += loss.item() * n
            total_tok += n
            
    return total_loss / max(total_tok, 1)

def train_model(mname, vocab, train_names, val_names, args, device):
    print(f"\n{'='*62}\n  Training: {mname.upper()}\n{'='*62}")
    train_ds     = NamesDataset(train_names, vocab)
    val_ds       = NamesDataset(val_names, vocab)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model     = build_model(mname, len(vocab)).to(device)
    hp        = HPARAMS[mname]
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = nn.NLLLoss(ignore_index=0)

    print(f"  Params: {model.count_parameters():,}  |  Train/Val: {len(train_ds)}/{len(val_ds)}\n")

    best_val, best_path, no_improve = float("inf"), os.path.join(args.ckpt_dir, f"{mname}.pt"), 0

    for epoch in range(1, args.epochs + 1):
        tr  = run_epoch(model, train_loader, optimizer, criterion, device, mname, True)
        val = run_epoch(model, val_loader,   optimizer, criterion, device, mname, False)
        scheduler.step()

        if val < best_val:
            best_val = val
            torch.save(model.state_dict(), best_path)
            no_improve, flag = 0, "✓"
        else:
            no_improve += 1; flag = " "

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{args.epochs}  train={tr:.4f}  val={val:.4f}  {flag}")

        if no_improve >= args.patience:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    print(f"\n  Best val: {best_val:.4f}")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",  default=DEFAULTS["data_path"])
    parser.add_argument("--epochs",     type=int,   default=DEFAULTS["epochs"])
    parser.add_argument("--batch_size", type=int,   default=DEFAULTS["batch_size"])
    parser.add_argument("--val_split",  type=float, default=DEFAULTS["val_split"])
    parser.add_argument("--model",      default=DEFAULTS["model"],
                        choices=["all", "vanilla_rnn", "blstm_old", "blstm_new", "rnn_attention_old", "rnn_attention_new"])
    parser.add_argument("--ckpt_dir",   default=DEFAULTS["ckpt_dir"])
    parser.add_argument("--seed",       type=int,   default=DEFAULTS["seed"])
    parser.add_argument("--patience",   type=int,   default=DEFAULTS["patience"])
    args = parser.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_names = load_names(args.data_path)
    vocab = Vocabulary(); vocab.build(all_names)
    print(f"Loaded {len(all_names)} names | Vocab: {len(vocab)} chars")
    
    # Save vocab for the evaluate.py script
    torch.save(vocab, os.path.join(args.ckpt_dir, "vocab.pt"))

    random.shuffle(all_names)
    val_size    = max(1, int(len(all_names) * args.val_split))
    val_names   = all_names[:val_size]
    train_names = all_names[val_size:]
    print(f"Train: {len(train_names)}  |  Val: {len(val_names)}\n")

    if args.model == "all":
        model_names = ["vanilla_rnn", "blstm_old", "blstm_new", "rnn_attention_old", "rnn_attention_new"]
    else:
        model_names = [args.model]

    trained = {}
    for name in model_names:
        trained[name] = train_model(name, vocab, train_names, val_names, args, device)

    print("\n" + "="*62 + "\n  All training complete!\n" + "="*62)
    
    sos = vocab.char2idx["<SOS>"]
    eos = vocab.char2idx["<EOS>"]
    
    print("\nSample names (temperature=0.8):")
    for name, model in trained.items():
        try:
            samples = [vocab.decode(model.generate(sos, eos, temperature=0.8, device=device))
                       for _ in range(8)]
            print(f"  {name:<20}: {samples}")
        except Exception as e:
            # The old models might crash during generation due to architectural flaws.
            # We catch it here so it doesn't break the script, which is perfect for your report!
            print(f"  {name:<20}: [GENERATION FAILED - Architectural incompatibility]")

if __name__ == "__main__":
    main()