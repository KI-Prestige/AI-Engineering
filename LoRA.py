import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class LoRALinear(nn.Module):
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Original linear layer (will be frozen)
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Freeze the original weight
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original path
        result = self.linear(x)

        # LoRA path
        lora_out = self.lora_dropout(x) @ self.lora_A.T   # (..., rank)
        lora_out = lora_out @ self.lora_B.T               # (..., out_features)
        result = result + lora_out * self.scaling

        return result

    def merge_weights(self):
        
        if self.rank > 0:
            delta = (self.lora_B @ self.lora_A) * self.scaling
            self.linear.weight.data += delta
            # Optionally zero out adapters after merge
            # self.lora_A.data.zero_()
            # self.lora_B.data.zero_()


def replace_linear_with_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
    target_modules: Optional[list] = None,
):
   
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
            # Create LoRA version
            lora_linear = LoRALinear(
                in_features=module.in_features,
                out_features=module.out_features,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                bias=module.bias is not None,
            )
            # Copy original weights
            lora_linear.linear.weight.data = module.weight.data.clone()
            if module.bias is not None:
                lora_linear.linear.bias.data = module.bias.data.clone()

            setattr(model, name, lora_linear)
        else:
            # Recurse
            replace_linear_with_lora(module, rank, alpha, dropout, target_modules)

    return model


from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Meta-Llama-3-8B"   # or any decoder-only model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Apply LoRA
model = replace_linear_with_lora(
    model,
    rank=16,
    alpha=32,
    dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# Only train LoRA parameters
for name, param in model.named_parameters():
    if "lora_" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Print trainable params
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")



from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

optimizer = AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=2e-4,          # LoRA likes higher LR than full FT
    weight_decay=0.0, # usually no weight decay on adapters
)

# Standard causal LM training loop
model.train()
for batch in dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()