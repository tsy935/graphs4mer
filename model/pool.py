"""
From https://github.com/HazyResearch/state-spaces
Authors: albertfgu & krandiash
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DownAvgPool(nn.Module):

    def __init__(self, d_input, stride=1, expand=1, transposed=True):
        super().__init__()
        self.d_input = d_input
        self.stride = stride
        self.expand = expand
        self.transposed = transposed

    def forward(self, x):
        if not self.transposed:
            x = rearrange(x, "b ... d -> b d ...")
        # einops appears slower than F
        if x.ndim == 3:
            x = F.avg_pool1d(x, self.stride, self.stride)
        elif x.ndim == 4:
            x = F.avg_pool2d(x, self.stride, self.stride)
        else:
            # Reduction string e.g. "b d (l1 2) (l2 2) -> b d l1 l2"
            reduce_str = (
                "b d "
                + " ".join([f"(l{i} {self.stride})" for i in range(x.ndim - 2)])
                + " -> b d "
                + " ".join([f"l{i}" for i in range(x.ndim - 2)])
            )
            x = reduce(x, reduce_str, "mean")

        if self.expand > 1:
            x = repeat(x, "b d ... -> b (d e) ...", e=self.expand)
        if not self.transposed:
            x = rearrange(x, "b d ... -> b ... d")
        return x

    def step(self, x, state, **kwargs):
        if self.stride > 1 or self.expand > 1:
            raise NotImplementedError
        return x, state

    @property
    def d_output(self):
        return self.d_input * self.expand
