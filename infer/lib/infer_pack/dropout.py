from typing import Any, Mapping
from torch import Tensor
import torch.nn as nn
import torch
class Dropouts(nn.Dropout):
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super(Dropouts, self).__init__(p, inplace)
        self.is_eval= False
        self.drop = nn.Dropout(p, inplace)

    def forward(self, input: Tensor) -> Tensor:
        return self.drop.forward(input) if not self.is_eval else input

        
    # def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
    #     return self.drop.load_state_dict(state_dict, strict)
    
    @torch.jit.ignore
    def eval(self):
        self.is_eval = True
        return super().eval()