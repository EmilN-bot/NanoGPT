import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size=64
block_size=256
max_iters=2500
eval_interval=500
learning_rate = 3e-4
device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
eval_iters=200
n_embd = 384
n_head = 6
n_layer = 3 # takes forever to train, potentially reduce this initially 6
dropout = 0.2
#-----------

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf8') as f:
    text = f.read()

#determine all characters
chars = sorted(list(set(text)))
vocab_size = len(chars)
#create mapping of characters to numbers (tokenizer)
stoi = { ch:i for i, ch in enumerate(chars)}
itos = { i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] #encodes characters to numbers
decode = lambda s: ''.join([itos[c] for c in s]) #decodes numbers to characters
# there are many other different tokenizers but this is just for simple purpose

#encode entire dataset
data = torch.tensor(encode(text), dtype=torch.long)
#seperate into train and val set
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    #returns batches based on batchsize and blocksize
    data = train_data if split=='train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1]for i in ix])
    x, y = x.to(device), y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss():
    out={}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """one head of self attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) #(B,T,C)
        q = self.query(x) #(B,T,C)
        #compute attention scores (affinities)
        wei = q @ k.transpose(-2,-1) * C**-0.5 #(B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) #(B,T,T)
        wei = self.dropout(wei)
        #perform weighted aggregation of the values
        v = self.value(x) #(B,T,C)
        out = wei @ v 
        return out

class MulitHeadAttention(nn.Module):
    """Multiple heads of self attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))

        return out

class FeedForward(nn.Module):
    """simple linear layer followed by non linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd,4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # projection layer
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer Block: communication followed by computation"""

    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_embd//n_heads
        self.sa = MulitHeadAttention(n_heads, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_heads=4) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) #final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensors of integers
        tok_emb = self.token_embedding_table(idx) #(B,T,n_embed)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) #(T,C)
        x = tok_emb + pos_emb #(B,T,C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  #(B,T,vocab_size)
        if targets==None:
            loss=None
        else:
            B, T, C = logits.shape
            logits= logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        #idx is (B,T) Array of the current context
        for _ in range(max_new_tokens):
            #crop idx to the last block_size token
            idx_cond = idx[:, -block_size:]
            #get the predictions
            logits, loss = self(idx_cond)
            #focus only on the last time step
            logits = logits[:, -1, :] #becomes (B,C)
            #apply softmax to get probabillities
            probs = F.softmax(logits, dim=1) #(B,C)
            #sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) #(B,1)
            #append samples index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) #(B,T+1)
        return idx


    
model = BigramLanguageModel()
m = model.to(device)

#create a pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for iter in range(max_iters):


    #every once in a while evaluate loss on trainset and val set
    if iter%eval_interval==0:
        losses = estimate_loss()
        print(f'step {iter}: train loss {losses['train']:.4f}, val loss{losses['val']:.4f}')

    #sample batch of data train
    xb, yb = get_batch('train')

    #evaluate their loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#generate from modell
print(decode(m.generate(idx = torch.zeros((1,1), dtype=torch.long, device=device), max_new_tokens=1500)[0].tolist()))


