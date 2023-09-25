from typing import Any, Mapping
from torch import Tensor
import torch.nn as nn
import torch


class Dropout(nn.Dropout):
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super(Dropout, self).__init__(p, inplace)
        self.is_eval= False
        self.drop = nn.Dropout(p, inplace)

    def forward(self, input: Tensor) -> Tensor:
        return self.drop.forward(input) if not self.is_eval else input
    
    @torch.jit.ignore
    def eval(self):
        self.is_eval = True
        return super().eval()