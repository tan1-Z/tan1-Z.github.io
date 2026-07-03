---
title: model_minimind
date: 2026-07-03 12:00:00
tags:
  - Blog
categories:
  - Study Note
  - MiniMindModel
katex: true
---

# Minimindconfig

```python
self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
```

推理时上下文外推。也就是模型训练时可能主要见过较短上下文，但推理时希望它处理更长上下文。
这里用的是 YaRN 思路：它会调整 RoPE 的频率，让模型在更长位置上不至于位置编码完全失真。

使模型支持更长输入，但不等同于具有相同质量的长文本理解能力

# Attention

## RMSNorm

Layer Norm：(x - mean) / sqrt(var + eps)

RMSNorm： x / sqrt(mean(x^2) + eps)    更快

```python
class RMSNorm(nn.Module):
		def __init__(self, dim, eps=1e-5):
				super().__init__()
				self.eps = eps
				self.weight = nn.Parameter(torch.ones(dim))
				
		def norm(self, x):
				return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
		
		def forward(self, x):
				return (self.weight * self.norm(x.float())).type_as(x)
```

```python
class LayerNorm(nn.Module):
		def __init__(self, dim, eps=1e-5):
				self.eps = eps
				self.weight = nn.Parameter(torch.ones(dim))
				self,bias = nn.Parameter(torch.zeros(dim))
				
		def forward(self, x):
				mean = x.mean(dim=-1, keepdim=True)
				var = x.var(dim=-1, keepdim=True, unbiased=False)
				x_norm = (x - mean) * torch.rsqrt(var + self.eps)
				return self.weight * x_norm + self.bias
```

## RoPE频率计算

计算Rotary Position Embedding所需的cos和sin表

```python
def precompute_freq_cis(dim, end=int(32 * 1024), rope_base=1e6, rope_scaling: dict=None):
		freqs, attn_factor = 1.0 / (rope_base ** torch.arange(0, dim, 2)[: (dim//2)].float(), 1.0
		if rope_scaling is not None:
				orig_max, factor, beta_fast, beta_slow, attn_factor = (
						rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
						rope_scaling.get("beta_fast", 32.0), rope_scaling,get("beta_slow", 1.0),
						rope_scaling.get("attention_factor", 1.0)
				)
				if end / org_max > 1.0:   # 当前长度大于原始长度则需要外推
						inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
						low, high = (
								max(math.floor(inv_dim(beta_fast)), 0),
								min(math.ceil(inv_dim(beta_slow)), dim // 2 -1)
						)
```

freq_i = 1 / base ^ (2i / dim)

YaRN：f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp

`orig_max`： 原始上下文训练长度参考值

`factor`：表示希望外推到更长范围

`beta_fast`/`beta_slow`：控制哪些频率维度更快/更慢过渡

外推的思路：

把一部分 freqs 变小
=> 同样的位置 position 对应更小的旋转角度
=> 长位置被“压缩”到模型更熟悉的旋转范围里

频率freq_i对应的波长：wavelength_i = 2π / freq_i= 2π * rope_base^(2i / dim)
找到“波长约等于 `orig_max / b`”的那个维度，则解方程：
2π * rope_base^(2i / dim) = orig_max / b
得到i = dim * log(orig_max / (b * 2π)) / (2 * log(rope_base))，即`inv_dim`

计算过渡区间：
low：快频率区域的边界
high：慢

低维度：频率高，波长短，擅长局部位置
高维度：频率低，波长长，擅长长距离位置
