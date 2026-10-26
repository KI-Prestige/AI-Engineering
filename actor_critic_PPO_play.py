import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared= nn.Linear(1, 8)
        self.policy= nn.Linear(8, 2)
        self.value= nn.Linear(8, 1)

    def forward(self, x):
        h= torch.tanh(self.shared(x))
        return self.policy(h), self.value(h)


model= ActorCritic()
opt= torch.optim.Adam(model.parameters(), lr=0.05)

true_rewards= [1.0, 10.0]
EPS= 0.2
old_logprobs= None

for episode in range(80):
    state= torch.zeros(1, 1)

    logits, value= model(state)
    log_probs= F.log_softmax(logits, dim=-1)
    probs= log_probs.exp()

    action= torch.multinomial(probs, 1).item()
    reward= true_rewards[action] + 0.3 * torch.randn(1).item()

    advantage= reward - value.item()

    if old_logprobs is None:
        ratio= torch.tensor(1.0)
    else:
        ratio= (log_probs[0, action] - old_logprobs[action]).exp()

    unclipped= ratio * advantage
    clipped= torch.clamp(ratio, 1-EPS, 1+EPS) * advantage
    policy_loss = -torch.min(unclipped, clipped)

    value_loss= (value - reward)**2

    loss= policy_loss + 0.5 * value_loss

    opt.zero_grad()
    loss.backward()
    opt.step()

    old_logprobs= log_probs.detach()[0]

    if episode % 10 == 0:
        with torch.no_grad():
            p= F.softmax(model(state)[0], dim=-1)[0]
            v= model(state)[1].item()
        print(f"ep {episode:2d} act={action} r={reward:5.1f}   " 
              f"A= {advantage:5.1f} V={v:5.1f}  "
              f"P0={p[0]:.2f} P1={p[1]:.2f}     ratio={ratio.item():.2f}")