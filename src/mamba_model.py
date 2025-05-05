# Mamba components adapted from a minimal version from https://github.com/johnma2006/mamba-minimal
# Includes ModelArgs, MambaBlock, ResidualBlock, RMSNorm

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from einops import rearrange, repeat, einsum
from typing import Union  # Keep for potential internal use, but config input is string


@dataclass
class ModelArgs:
    d_model: int
    n_layer: (
        int  # Not directly used by MambaBlock/ResidualBlock but kept for consistency
    )
    # vocab_size: int # Not needed for ICL integration
    d_state: int = 16
    expand: int = 2
    # dt_rank is received as a string ("auto" or integer string) from config
    dt_rank: str = "auto"
    d_conv: int = 4
    # pad_vocab_size_multiple: int = 8 # Not needed
    conv_bias: bool = True
    bias: bool = False

    # Fields added during __post_init__ in original, calculated here directly
    d_inner: int = field(init=False)
    dt_rank_value: int = field(init=False)  # Store the calculated integer dt_rank

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)

        # Parse dt_rank string
        if self.dt_rank == "auto":
            self.dt_rank_value = math.ceil(self.d_model / 16)
        else:
            try:
                # Attempt to convert the string to an integer
                dt_rank_int = int(self.dt_rank)
                if dt_rank_int <= 0:
                    raise ValueError(
                        "dt_rank must be positive if specified as an integer string"
                    )
                self.dt_rank_value = dt_rank_int
            except ValueError:
                raise ValueError(
                    f"dt_rank must be an integer string or 'auto', got 	{self.dt_rank!r}"
                )
        # vocab_size padding not needed for ICL


class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super().__init__()
        self.args = args

        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)

        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        # Use the calculated integer dt_rank_value here
        self.x_proj = nn.Linear(
            args.d_inner, args.dt_rank_value + args.d_state * 2, bias=False
        )

        # dt_proj projects Δ from dt_rank to d_in
        # Use the calculated integer dt_rank_value here
        self.dt_proj = nn.Linear(args.dt_rank_value, args.d_inner, bias=True)

        A = repeat(torch.arange(1, args.d_state + 1), "n -> d n", d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def forward(self, x):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].

        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d)

        """
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(
            split_size=[self.args.d_inner, self.args.d_inner], dim=-1
        )

        x = rearrange(x, "b l d_in -> b d_in l")
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, "b d_in l -> b l d_in")

        x = F.silu(x)

        y = self.ssm(x)

        y = y * F.silu(res)

        output = self.out_proj(y)

        return output

    def ssm(self, x):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d_in)

        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)

        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank_value + 2*n)

        # Use the calculated integer dt_rank_value for splitting
        (delta, B, C) = x_dbl.split(
            split_size=[self.args.dt_rank_value, n, n], dim=-1
        )  # delta: (b, l, dt_rank_value). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)

        y = self.selective_scan(
            x, delta, A, B, C, D
        )  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]

        return y

    def selective_scan(self, u, delta, A, B, C, D):
        """Does selective scan algorithm. See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).

        Args:
            u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)

        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
            Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.

        """
        (b, l, d_in) = u.shape
        n = A.shape[1]

        # Discretize continuous parameters (A, B)
        # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
        # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
        #   "A is the more important term and the performance doesn't change much with the simplification on B"
        deltaA = torch.exp(einsum(delta, A, "b l d_in, d_in n -> b l d_in n"))
        deltaB_u = einsum(delta, B, u, "b l d_in, b l n, b l d_in -> b l d_in n")

        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        # Note that the below is sequential, while the official implementation does a much faster parallel scan that
        # is additionally hardware-aware (like FlashAttention).
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], "b d_in n, b n -> b d_in")
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)

        y = y + u * D

        return y


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()
        self.args = args
        self.mixer = MambaBlock(args)
        self.norm = RMSNorm(args.d_model)

    def forward(self, x):
        """
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d)
        """
        # Implementation follows the paper's block structure: Norm -> Mixer -> Add
        output = self.mixer(self.norm(x)) + x
        return output


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # The x.to(torch.float32) is crucial for stability, especially with lower precision inputs.
        output = (
            x.to(torch.float32)
            * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            * self.weight
        )
        # Cast back to original dtype
        return output.to(x.dtype)
