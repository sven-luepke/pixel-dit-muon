from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange


BASE_WIDTH = 256  # Base width for muP scaling


@dataclass
class ModelConfig:
    d_model: int
    d_ff: int
    n_heads: int
    n_layers: int
    image_size: tuple[int, int]
    patch_size: int
    bottleneck_size: int
    num_classes: int
    min_freq: float
    max_freq: float


class RoPEEmbedder(nn.Module):
    """
    Golden Gate RoPE based on https://jerryxio.ng/posts/nd-rope/
    """
    def __init__(self, dim: int, n_heads: int, min_freq: float, max_freq: float):
        super().__init__()
        self.n_heads = n_heads
        self.n_freqs = dim // 2

        # Log-spaced frequency magnitudes with explicit min/max bounds.
        omega = min_freq * (max_freq / min_freq) ** torch.linspace(0, 1, self.n_freqs)
        direction_spacing = math.pi * (math.sqrt(5.0) - 1.0)  # pi / phi
        phi_base = torch.arange(self.n_freqs, dtype=omega.dtype).unsqueeze(0) * direction_spacing
        phi_shift = (
            torch.arange(n_heads, dtype=omega.dtype).unsqueeze(1) * (direction_spacing / n_heads)
        )
        phi = phi_base + phi_shift
        directions = torch.stack((torch.cos(phi), torch.sin(phi)), dim=-1)
        self.register_buffer("freqs_hfp", directions * omega.unsqueeze(0).unsqueeze(-1))

    def forward(self, pos: Tensor) -> Tensor:
        phase = torch.einsum("...p,hfp->...hf", pos, self.freqs_hfp.to(pos.device, pos.dtype))
        emb = torch.stack(
            [torch.cos(phase), -torch.sin(phase), torch.sin(phase), torch.cos(phase)],
            dim=-1,
        )
        return rearrange(emb, "b n h f (i j) -> b h n f i j", i=2, j=2)


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


class FeedForward(nn.Module):

    def __init__(self, config: ModelConfig):

        super().__init__()
        self.ff_in_proj = nn.Linear(config.d_model, 2 * config.d_ff, bias=False)
        self.ff_out_proj = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ff_in_proj(x)
        x, gate = x.chunk(2, dim=-1)
        x = x * F.silu(gate)
        return self.ff_out_proj(x)


class SelfAttention(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads

        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)

        # QK-norm
        head_dim = config.d_model // config.n_heads
        self.q_norm = nn.RMSNorm(head_dim, elementwise_affine=False)
        self.k_norm = nn.RMSNorm(head_dim, elementwise_affine=False)

        self.attn_out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, "b s (qkv h d) -> qkv b s h d", qkv=3, h=self.n_heads)

        # QK-Norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Attention
        q, k, v = (rearrange(t, "b s h d -> b h s d") for t in (q, k, v))
        q, k = apply_rope(q, k, pe)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = rearrange(attn_out, "b h s d -> b s (h d)")

        return self.attn_out_proj(attn_out)


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


class DiTBlock(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.norm_attn = nn.LayerNorm(config.d_model, elementwise_affine=False)
        self.self_attention = SelfAttention(config)
        self.norm_ff = nn.LayerNorm(config.d_model, elementwise_affine=False)
        self.feed_forward = FeedForward(config)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.d_model, 6 * config.d_model, bias=False)
        )

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor):
        shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff = self.adaLN_modulation(vec).unsqueeze(1).chunk(6, dim=-1)
        x = x + self.self_attention(modulate(self.norm_attn(x), shift_sa, scale_sa), pe) * gate_sa
        x = x + self.feed_forward(modulate(self.norm_ff(x), shift_ff, scale_ff)) * gate_ff
        return x


class FinalLayer(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.output_scale = BASE_WIDTH / config.d_model
        self.norm = nn.LayerNorm(config.d_model, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(config.d_model, config.patch_size * config.patch_size * 3, bias=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.d_model, 2 * config.d_model, bias=False)
        )

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).unsqueeze(1).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x) * self.output_scale


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, max_period=10000):
        super().__init__()
        self.input_proj = nn.Linear(frequency_embedding_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.frequency_embedding_size = frequency_embedding_size
        half = frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        )
        self.register_buffer("freqs", freqs)

    def timestep_embedding(self, t: Tensor, time_factor: float = 1000.0):
        """
        Create sinusoidal timestep embeddings.
        :param t: a Tensor of arbitrary shape of (possibly fractional) timesteps.
        :return: a Tensor of shape (*t.shape, dim) of positional embeddings.
        """
        t = time_factor * t
        args = t.float().unsqueeze(-1) * self.freqs
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding.to(t.dtype)

    def forward(self, t):
        t_freq = self.timestep_embedding(t)
        return self.out_proj(F.silu(self.input_proj(t_freq)))


class DiT(nn.Module):

    """
    Modern Diffusion Transformer with:
    - RoPE positional encoding
    - QK-Norm
    - No biases in linear layers
    - SwiGLU activation (FeedForward)
    """
    def __init__(self, config: ModelConfig):

        super().__init__()
        self.config = config
        self.timestep_embedder = TimestepEmbedder(config.d_model)
        self.class_embedder = nn.Embedding(config.num_classes, config.d_model)
        self.patch_embed = nn.Sequential(
            nn.Linear(3 * config.patch_size * config.patch_size, config.bottleneck_size, bias=False),
            nn.Linear(config.bottleneck_size, config.d_model, bias=False),
        )

        # RoPE positional encoding
        head_dim = config.d_model // config.n_heads
        self.rope_embedder = RoPEEmbedder(
            dim=head_dim,
            n_heads=config.n_heads,
            min_freq=config.min_freq,
            max_freq=config.max_freq,
        )

        self.blocks = nn.ModuleList(
            [DiTBlock(config) for _ in range(config.n_layers)]
        )
        self.final_layer = FinalLayer(config)

        self.init_weights()

    def init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                fan_in = module.weight.shape[1]
                torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(1.0 / fan_in))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # zero-init output layer
        nn.init.constant_(self.final_layer.linear.weight, 0)

    def forward(self, x, x_pos, t, c):
        _, C, H, W = x.shape
        p = self.config.patch_size
        H_grid, W_grid = H // p, W // p

        # patchify
        x = rearrange(x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", h=H_grid, w=W_grid, p1=p, p2=p)
        x = self.patch_embed(x)

        # condition time and class embeddings
        t_emb = self.timestep_embedder(t)
        c_emb = self.class_embedder(c)
        vec = t_emb + c_emb

        pe = self.rope_embedder(x_pos)

        for block in self.blocks:
            x = block(x, vec, pe)
        x = self.final_layer(x, vec)

        # unpatchify
        return rearrange(x, "b (h w) (c p1 p2) -> b c (h p1) (w p2)", h=H_grid, w=W_grid, c=C, p1=p, p2=p)


def create_model(n_heads: int, n_layers: int):
    d_head = 128
    d_model = d_head * n_heads
    d_ff = math.ceil(((8 / 3) * d_model) / 128) * 128
    config = ModelConfig(
        d_model=d_model,
        d_ff=d_ff,
        n_heads=n_heads,
        n_layers=n_layers,
        image_size=(256, 256),
        patch_size=16,
        bottleneck_size=128,
        num_classes=1000,
        min_freq=0.1,
        max_freq=16,
    )
    return DiT(config)
