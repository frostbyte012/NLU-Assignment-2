import os,torch
from data_utils import load_names,Vocabulary
from generate_utils import sample_next 
from train import build_model,HPARAMS


def calculate_metrics(generated_names, original_dataset) :
    """Calculates Novelty and Diversity for Task-2"""
    original_set=set([n.lower() for n in original_dataset])
    generated_set=set([n.lower() for n in generated_names if len(n)>2]) 
    
    if not generated_set :
        return 0.0, 0.0
        
    novel_names=generated_set-original_set
    
    novelty_rate=len(novel_names)/len(generated_names)
    diversity=len(generated_set)/len(generated_names)
    
    return novelty_rate*100,diversity*100

def main() :
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_names=load_names("TrainingNames.txt")
    vocab=torch.load("checkpoints/vocab.pt",weights_only=False)
    
    models_to_eval=["vanilla_rnn","blstm_old","blstm_new","rnn_attention_old","rnn_attention_new"]
    
    print(f"{'Model':<20}|{'Novelty Rate (%)':<18}|{'Diversity (%)':<15}")
    print("-"*60)
    
    for mname in models_to_eval :
        model=build_model(mname,len(vocab)).to(device)
        ckpt_path=f"checkpoints/{mname}.pt"
        
        if not os.path.exists(ckpt_path) :
            print(f"{mname:<20} | Model checkpoint not found.")
            continue
            
        model.load_state_dict(torch.load(ckpt_path,map_location=device,weights_only=True))
        model.eval()
        
        generated=[]
        sos,eos=vocab.char2idx["<SOS>"],vocab.char2idx["<EOS>"]
        
        # Generateing 200 names per model
        
        for _ in range(200) :
            try :
                seq=model.generate(sos,eos,temperature=0.7,device=device)
                generated.append(vocab.decode(seq))
            except Exception as e :
                pass # If the old models crash, we just skip
                
        # The old model fails to generate 
        if len(generated)==0 :
            print(f"{mname:<20}|{'0.00':<18}|{'0.00':<15}(Failed Gen)")
            continue
            
        novelty,diversity=calculate_metrics(generated,original_names)
        print(f"{mname:<20}|{novelty:<18.2f}|{diversity:<15.2f}")

if __name__ == "__main__" :
    main()