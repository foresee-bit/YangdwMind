from transformers import PretrainedConfig
import torch
import torch.nn as nn
from typing import Optional
import math


# huggingface中的类，用于存储模型配置参数
class YangdwMindConfig(PretrainedConfig):
    model_type = "yangdwmind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.1,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 4,
                "beta_slow": 1,
                "factor": 4,
                "original_max_position_embeddings": 2048,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )


# 继承nn.Module类
class RMSNorm(nn.Module):
    # __init__初始化
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    # _norm
    def _norm(self, x):
        return torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    # forward
    # x.float() :将输入 x 转为 float32,避免在 float16 或 bfloat16 下计算 mean / sqrt 时出现数值下溢（underflow）或精度问题
    # .type_as(x): 将归一化结果转回原始输入的数据类型（如 x 是 torch.float16，就转回 float16）,保持模型整体精度一致
    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)


# RoPE
def precompute_freqs_cis(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: Optional[dict] = None,
):
    """
    Compute the cos and sin frequency tables for RoPE.

    Args:
        dim (int): Dimension of the model.
        end (int): Maximum sequence length.
        rope_base (float): Base frequency for RoPE.
        rope_scaling 这个参数可以是 dict 类型，也可以是 None，默认值就是 None
    """
    # 写出最初的RoPE式子
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 1),
            rope_scaling.get("beta_fast", 1),
            rope_scaling.get("beta_slow", 1),
        )
        # 计算 corr_dim,在 RoPE 的频率分量中，找到第一个其旋转周期超过原始最大上下文长度 orig_max 的维度索引 corr_dim。如果没有找到，则设为 dim // 2（即所有维度都不满足，全部需要处理）
        corr_dim = next(
            (i for i in range(2, dim // 2) if 2 * math.pi / freqs[i] > orig_max),
            dim // 2,
        )
        # 计算power
        power = torch.arange(0, dim // 2, device=freqs.device).float() / (
            max(dim // 2 - 1, 1)
        )
        # 计算beta
        beta = beta_slow + (beta_fast - beta_slow) * power

        # 计算scale
        scale = torch.where(
            torch.arange(dim // 2, device=freqs.device) < corr_dim,
            (beta * factor - beta + 1) / (beta * factor),
            1.0 / factor,
        )
        # 应用scale
        freqs = freqs * scale
    # 生成位置索引,与频率相乘
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()  # [end, dim // 2]

    # 返回一个cos和sin
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin
