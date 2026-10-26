import torch
import torch.nn as nn
import torch.nn.functional as F


class Tinytransformer(nn.Module):
    def __init__(self,vocab_size=8, n_embd=32, n_head=2, n_layer=2, block_size=8):
        super().__init__()
        self.block_size= block_size
        self.tok_emb= nn.Embedding(vocab_size, n_embd)
        self.pos_emb= nn.Embedding(block_size, n_embd)
        self.blocks= nn.ModuleList([
            nn.TransformerEncoderLayer(n_embd, n_head, dim_feedforward=64, batch_first=True)
            for _ in range(n_layer)
        ])
        self.ln_f= nn.LayerNorm(n_embd)
        self.head= nn.Linear(n_embd, vocab_size)
        self.value_head= nn.Linear(n_embd, 1)

    def forward(self, idx):
        B, T= idx.shape
        tok= self.tok_emb(idx)
        pos= self.pos_emb(torch.arange(T, device= idx.device))
        x= tok + pos
        for block in self.blocks:
            x= block(x)
        x= self.ln_f(x)
        logits= self.head(x)
        values= self.value_head(x)[:, -1]
        return logits, values


def sample_group(model, prompt, G=4, max_new=3, temperature=1.0):
    model.eval()
    completions, logprobs_list= [], []
    with torch.no_grad():
        for _ in range(G):
            idx= prompt.clone()
            logp_sum= 0.0
            for _ in range(max_new):
                logits, _= model(idx)
                logits= logits[:, -1, :]/ temperature
                probs= F.softmax(logits, dim=-1)
                next_token= torch.multinomial(probs, 1)
                logp_sum+= torch.log(probs[0, next_token.item()] + 1e-8)
                idx= torch.cat([idx, next_token], dim=1)
            completions.append(idx)
            logprobs_list.append(logp_sum)
    model.train()
    return completions, torch.stack(logprobs_list)

def grpo_advantages(rewards):
    mean= rewards.mean()
    std= rewards.std() + 1e-4
    return (rewards - mean) / std


VOCAB= 8
CORRECT_TOKEN= 5

def reward_fn(completion):
    return 1.0 if completion[0, -1].item()== CORRECT_TOKEN else 0.0


model= Tinytransformer(vocab_size=VOCAB)
opt= torch.optim.AdamW(model.parameters(), lr= 3e-8)
prompt= torch.tensor([[1, 2, 3]])
G= 4
EPS= 0.2


for step in range(60):
    completions, old_logprobs= sample_group(model, prompt, G=G)

    rewards= torch.tensor([reward_fn(c) for c in completions], dtype=torch.float)

    advantages= grpo_advantages(rewards)

    new_logprobs= []
    for c in completions:
        logits, _ = model(c[:, :-1])
        logp= F.log_softmax(logits, dim=-1)
        gen_tokens= c[:, 1:]
        lp= torch.gather(logp, 2, gen_tokens.unsqueeze(-1)).squeeze(-1)
        new_logprobs.append(lp.sum())
    new_logprobs= torch.stack(new_logprobs)

    ratio= (new_logprobs - old_logprobs.detach()).exp()
    unclipped= ratio * advantages
    clipped= torch.clamp(ratio, 1-EPS, 1+EPS) * advantages
    loss= torch.min(unclipped, clipped).mean()

    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 10 == 0:
            print(f"step {step:2d}  rewards={rewards.tolist()}  "
                  f"adv={advantages.tolist()}  "
                  f"loss={loss.item():.3f}")