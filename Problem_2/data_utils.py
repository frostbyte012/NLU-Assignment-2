"""
Handling vocabulary construction and dataset preparation
for character-level name generation.
"""

#imports fot data
import torch
from torch.utils.data import Dataset


# ----Special-Tokens---
PAD_TOKEN="<PAD>"   #batch padding
SOS_TOKEN="<SOS>"   #the start of sequence
EOS_TOKEN="<EOS>"   #End Seq padding


class Vocabulary:
    

    def __init__(self) :
        self.char2idx={PAD_TOKEN:0,SOS_TOKEN:1,EOS_TOKEN:2}
        self.idx2char={0:PAD_TOKEN,1:SOS_TOKEN,2:EOS_TOKEN}
        self.n_chars=3   #Special Tokens

    def build(self, names: list[str])->None :
        """Populating the vocab from a list of name strings."""
        for name in names :
            for ch in name :
                if ch not in self.char2idx :
                    self.char2idx[ch]=self.n_chars
                    self.idx2char[self.n_chars]=ch
                    self.n_chars+=1

    def encode(self,name:str)->list[int]:
       
        return (
            [self.char2idx[SOS_TOKEN]]
            + [self.char2idx[ch] for ch in name]
            + [self.char2idx[EOS_TOKEN]]
        )

    def decode(self,indices:list[int])->str :
        """Convert index list→string,skipping special tokens."""
        return "".join(
            self.idx2char[i]
            for i in indices
            if i not in (0, 1, 2)            # skip PAD / SOS / EOS
        )

    def __len__(self)->int:
        return self.n_chars


class NamesDataset(Dataset) :
    """
        input→[SOS,c1,c2,…,cN]
        target→[c1,c2,…,cN,EOS]

    This matches a next-character prediction objective: given every
    prefix the model must predict the next character.
    """

    def __init__(self,names:list[str],vocab:Vocabulary) :
        self.vocab=vocab
        self.samples=[]

        for name in names :
            encoded=vocab.encode(name) #[SOS, …chars…, EOS]
            inp=torch.tensor(encoded[:-1],dtype=torch.long)
            target=torch.tensor(encoded[1:],dtype=torch.long)
            self.samples.append((inp,target))

    def __len__(self)->int:
        return len(self.samples)

    def __getitem__(self,idx:int):
        return self.samples[idx]


def collate_fn(batch):
    """
    Custom collate: pads variable-length sequences to the same length
    within a mini-batch.

    Returns
    -------
    inputs  : (B, T_max)  LongTensor – padded input sequences
    targets : (B, T_max)  LongTensor – padded target sequences
    lengths : (B,)        LongTensor – original (unpadded) lengths
    """
    inputs,targets=zip(*batch)
    lengths=torch.tensor([len(x) for x in inputs],dtype=torch.long)

    inputs=torch.nn.utils.rnn.pad_sequence(inputs,batch_first=True,padding_value=0)
    targets=torch.nn.utils.rnn.pad_sequence(targets,batch_first=True,padding_value=0)

    return inputs,targets,lengths


def load_names(filepath:str)->list[str]:
    with open(filepath,"r",encoding="utf-8") as f:
        names=[line.strip() for line in f if line.strip()]
    return names
