import torch


def sample_next(log_probs_1d: torch.Tensor,
                temperature: float=0.8,
                top_k: int=10,
                repetition_penalty: float=1.5,
                generated_so_far: list=None,
                eos_idx: int=2,
                min_length_reached: bool=True)->int :
    """
    Nucleus style sampling with repetition penalty.
    """
    logits=log_probs_1d.clone().float()

    # Hard-mask PAD and SOS
    logits[0]=float('-inf')
    logits[1]=float('-inf')

    # Suppress EOS until minimum length
    if not min_length_reached :
        logits[eos_idx]=float('-inf')

    # Repetition penalty 
    if generated_so_far :
        for idx in set(generated_so_far) : 
            if logits[idx] != float('-inf') :
                # Applying a strong penalty if the character used
                if idx==generated_so_far[-1]:
                    logits[idx]=logits[idx]/(repetition_penalty*1.5)
                else:
                    logits[idx]=logits[idx]/repetition_penalty

    #Temperature
    logits=logits/max(temperature,1e-8)

    #Top-k filter
    top_vals,top_idx=torch.topk(logits,min(top_k,int((logits>float('-inf')).sum())))
    filtered=torch.full_like(logits,float('-inf'))
    filtered.scatter_(0,top_idx,top_vals)

    probs=torch.softmax(filtered,dim=-1)
    return  torch.multinomial(probs,1).item()
