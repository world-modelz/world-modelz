from copy import deepcopy
import torch
import torch.nn as nn


class EmaTraining(nn.Module):
    def __init__(self, model: nn.Module, decay: float=0.9999):
        super().__init__()
        self.decay = decay

        self.model = model
        self.shadow = deepcopy(self.model)

        for param in self.shadow.parameters():
            param.detach_()

    @torch.no_grad()
    def update(self):
        for targ, src in zip(self.shadow.parameters(), self.model.parameters()):
            targ.mul_(self.decay).add_(src, alpha=1 - self.decay)

        for targ, src in zip(self.shadow.buffers(), self.model.buffers()):
            targ.copy_(buffer)

    def forward(self, *input, **kwargs):
        if self.training:
            return self.model(*input, **kwargs)
        else:
            return self.shadow(*input, **kwargs)
