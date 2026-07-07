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
				super().__init__()
				self.eps = eps
				self.weight = nn.Parameter(torch.ones(dim))
				self.bias = nn.Parameter(torch.zeros(dim))

		def forward(self, x):
				mean = x.mean(dim=-1, keepdim=True)
				var = x.var(dim=-1, keepdim=True, unbiased=False)
				x_norm = (x - mean) * torch.rsqrt(var + self.eps)
				return self.weight * x_norm + self.bias
```

## RoPE频率计算

计算Rotary Position Embedding所需的cos和sin表

```python
def precompute_freqs_cis(dim, end=int(32 * 1024), rope_base=1e6, rope_scaling: dict=None):
		freqs, attn_factor = 1.0 / (rope_base ** torch.arange(0, dim, 2)[: (dim//2)].float(), 1.0
		if rope_scaling is not None:
				orig_max, factor, beta_fast, beta_slow, attn_factor = (
						rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
						rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0),
						rope_scaling.get("attention_factor", 1.0)
				)
				if end / orig_max > 1.0:   # 当前长度大于原始长度则需要外推
						inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
						low, high = (
								max(math.floor(inv_dim(beta_fast)), 0),
								min(math.ceil(inv_dim(beta_slow)), dim // 2 -1)
						)
						ramp = torch.clamp(
								(torch.arange(dim//2, device=freqs.device) - low) / max(high - low, 0.001),
								0,
								1
						)
						freqs = freqs * (1- ramp + ramp / factor)
		t = torch.arange(end, device=freqs.device)
		freqs = torch.outer(t, freqs).float()
		freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
		freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
		return freqs_cos, freqs_sin
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

平滑过渡向量 ramp：

低于 low 的维度：ramp = 0
low 到 high：   ramp 从 0 逐渐变到 1
高于 high：     ramp = 1

```python
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
		def rotate_half(x):
				return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), dim=-1)
			q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
			k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
			return q_embed, k_embed
```

这里要注意的也就是cos和sin的维度变化：一开始二者维度为q, k: [B, S, H, D] ; cos, sin: [S, D]，unsqueeze后，cos, sin→ [S, 1, D]，与q, k相乘时再经广播机制变为[1, S, 1, D]

注意：这里不处理 `v`。RoPE 只作用在 Q/K 上

```python
def repeat_kv(x, n_rep):
		bs, seq_len, num_kv_heads, head_dim = x.shape
		if n_rep == 1: return x
		return (x[:,:,:,None,:].expand(bs, seq_len, num_kv_heads, n_rep, head_dim).reshape(bs, seq_len, num_kv_heads * n_rep, head_dim))
```

## Attention 模块代码

```python
class Attention(nn.Module):
		def __init__(self, config: MiniMindConfig):
				super().__init__()
				self.num_kv_heads = config.num_attention_heads if config.num_kv_heads is None else config.num_kv_heads
				self.n_loacl_heads = config.num_attention_heads
				self.n_local_kv_heads = self.num_kv_heads
				self.n_rep = self.n_local_heads // self.n_local_kv_heads
				self.head_dim = config.head_dim
				self.is_causal = True
				self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
				self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
				self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
				self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
				self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
				self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
				self.attn_dropout = nn.Dropout(config.dropout)
				self.resid_dropout = nn.Dropout(config.dropout)
				self.dropout = config.dropout
				self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

		def forward(self, x, position_embeddings, past_kv=None, use_cache=False, attention_mask=None):
				bs, seq_len, _ = x.shape
				xq, xk,xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
				xq = xq.view(bs, seq_len, self.n_local_heads, self.head_dim)
				xk = xk.view(bs, seq_len, self.n_local_kv_heads, self.head_dim)
				xv = xv.view(bs, seq_len, self.n_local_kv_heads, self.head_dim)
				xq, xk = self.q_norm(xq), self.k_norm(xk)
				cos, sin = position_embeddings
				xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
				if past_kv is not None:
						xk = torch.cat([past_kv[0], xk], dim=1)
						xv = torch.cat([past_kv[1], xv], dim=1)  # seq_len维度上拼接
				past_kv = (xk, xv) if ues_cache else None
				xq, xk, xv = (xq.transpose(1,2), repeat_kv(xk, self.n_rep).transpose(1,2), repeat_kv(xv, self.n_rep).transpose(1,2))
				if self.flash and (seq_len > 1) and (not self.causal or past_kv is None) and (attention_mask is None or torch.all(attention_mask==1)):
						output = F.scaled_dot_product_attention(xq,xk,xv, dropout=self.dropout if self.training else 0.0, i_causal=self.is_causal))
				else:
						scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
						if self.causal:
								scores[:,:,:, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
						if attention_mask is not None:
								scores += (1.0 - attention_mask) * 1e-9
						output = self.attn_drop(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv  # [bs, num_attention_heads, seq_len, head_dim]
				output = output.transpose(1,2).reshape(bs, seq_len, -1)
				output = self.resid_dropout(self.o_proj(output))
				return output, past_kv
```

`scores[:, :, :,  -seq_len:]`：表示最后一维的后seq_len列

`torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)`生成的上三角矩阵的情况：

[
[0,   -inf, -inf, -inf],
[0,    0,   -inf, -inf],
[0,    0,    0,   -inf],
[0,    0,    0,    0  ],
]

加到 scores 后，每个 token 就不能看未来 token

# FeedForward

```python
class FeedForward(nn.Module):
		def __init__(self, config: MiniMindConfig, intermediate_size: int=None):
				super().__init__()
				intermediate_size = intermediate_size or config.intermediate_size
				self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
				self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
				self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
				self.act_fn = ACT2FN[config.hidden_act]

		def forward(self, x):
				return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

![Universal Reasoning Model and ConvSwiGLU](/img/model-minimind-urm-convswiglu.png)

## MOEFeedForward

```python
class MOEFeedForward(nn.Module):
		def __init__(self, config: MiniMindConfig):
				super().__init__()
				self.config = config
				self.gate = nn.Linear(config.hidden_size, config.num_experts. bias=False)
				self.experts = nn.ModuleList([FeedForward(config, config.moe_intermediate_size) for _ in range(config.num_experts)])
				self.act_fn = ACT2FN[config.hidden_act]

		def forward(self, x):
				bs, seq_len, hidden_dim = x.shape
				x_flat = x.view(-1, hidden_dim)
				scores = F.softmax(self.gate(x_flat), dim=-1)  # [bs * seq_len, num_experts]
				topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
				if self.config.norm_topk_prob: topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
				y = torch.zeros_like(x_flat)
				for i, expert in enumerate(self.experts):
						mask = (topk_idx == i)  # 判断哪些token选择了这个expert i
						if mask.any():
								token_idx = mask.any(dim=-1).nonzero().flatten()
								weight = topk_weight[mask].view(-1, 1)  # [num_selected_tokens, 1]
								y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
						elif self.training:
								y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
				if self.training and self.config.router_aux_loss_coef > 0:
						load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)  # [top_k, num_experts]
						self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
				else:
						self.aux_loss = scores.new_zeros(1).squeeze()
					return y.view(bs, seq_len, hidden_dim)
```

`x_flat = x.view(-1, hidden_dim)` ：做这一步的目的就是为了方便后续一次性处理所有token

举个例子：

x_flat[0] = 原来的 x[0, 0, :]  # 样本0 token0
x_flat[1] = 原来的 x[0, 1, :]  # 样本0 token1
x_flat[2] = 原来的 x[0, 2, :]  # 样本0 token2
x_flat[3] = 原来的 x[1, 0, :]  # 样本1 token0
x_flat[4] = 原来的 x[1, 1, :]  # 样本1 token1
x_flat[5] = 原来的 x[1, 2, :]  # 样本1 token2

**对于scores，每一行概率分布的最大值所在的位置就代表某一样本的某一token应该走的expert对应的编号**

```text
topk_weight.shape = [B*S, K]
topk_idx.shape    = [B*S, K]
```

例如：

```text
scores      = [0.05, 0.80, 0.10, 0.05]
topk_idx    = [1]
topk_weight = [0.80]
```

如果 `num_experts_per_tok=2`，那每个 token 会选两个专家，然后把两个专家输出加权求和

举个例子。假设有 3 个 token、4 个专家：

```text
scores = [
    [0.10, 0.60, 0.20, 0.10],  # token 0
    [0.50, 0.10, 0.30, 0.10],  # token 1
    [0.05, 0.15, 0.10, 0.70],  # token 2
]
```

如果 `num_experts_per_tok=2`，`topk` 之后大概是：

```python
topk_idx = [
    [1, 2],  # token 0 选择 expert 1 和 expert 2
    [0, 2],  # token 1 选择 expert 0 和 expert 2
    [3, 1],  # token 2 选择 expert 3 和 expert 1
]
```

对应权重是：

```python
topk_weight = [
    [0.60, 0.20],
    [0.50, 0.30],
    [0.70, 0.15],
]
```

如果代码里：

```python
if self.config.norm_topk_prob:
    topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
```

那么每个 token 选中的专家权重会重新归一化：

```text
token 0: [0.60, 0.20] -> [0.75, 0.25]
token 1: [0.50, 0.30] -> [0.625, 0.375]
token 2: [0.70, 0.15] -> [0.8235, 0.1765]
```

最终 token 0 的输出就是：

```text
y_token0 = 0.75 * expert_1(x_token0) + 0.25 * expert_2(x_token0)
```

token1，token2同理

`y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))`

上面这行代码中，`x_flat[token_idx]`：[num_selected_tokens, hidden_dim]，经expert处理后对应的输出也是这个维度，再乘route权重，表示将topk个专家加权混合

`y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())`

具有工程意义的处理（from GPT）：如果某个专家在当前 batch 里没有任何 token 选中，那么它的参数没有参与前向计算。某些分布式训练或自动求导场景里，这可能导致“未使用参数”的问题，这段代码数值上等于 0，不影响输出。**但计算图上把这个专家的参数“挂”进来了，避免在梯度/分布式同步里出问题**

## MiniMindBlock

```python
class MiniMindBlock(nn.Module):
		def __init__(self, layer_id: int, config: MiniMindConfig):
				super().__init__()
				self.self_attn = Attention(config)
				self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
				self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
				self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

		def forward(self, hidden_states, position_embeddings, past_kv=None, use_cache=False, attention_mask=None):
				residual = hidden_states # [bs, seq_len, hidden_dim]
				hidden_states, present_kv = self.self_attn(
						self.input_layernorm(hidden_states), position_embeddings,
						past_kv, use_cache, attention_mask
				)
				hidden_states += resdual
				hidden_states += self.mlp(self.post_attention_layernorm(hidden_states))
				return hidden_states, present_kv
```

## MiniMindModel

```python
class MiniMindModel(nn.Module):
		def __init__(self, config: MiniMindConfig):
				super().__init__()
				self.config = config
				self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
				self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
				self.dropout = nn.dropout(config.dropout)
				self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(config.num_hidden_layers)])
				self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
				freqs_cos, freqs_sin = precompute_freqs_cis(
						dim=config.head_dim, end=config.max_position_embeddings,
						rope_base=config.rope_theta, rope_scaling=config.rope_scaling
				)
				self.register_buffer("freqs_cos", freqs_cos, persistent=False)
				self.register_buffer("freqs_sin", freqs_sin, persistent=False)

		def forward(self, input_ids, attention_mask=None, past_kv=None, use_cache=False, **kwargs):
				bs, seq_len = input_ids.shape
				if hasattr(past_kv, 'layers'): past_kv=None
				past_kv = past_kv or [None] * len(self.layers)
				start_pos = past_kv[0][0].shape[1] if past_kv[0] is not None else 0
				hidden_states = self.dropout(self.embed_tokens(input_ids))
				# Recompute RoPE buffers lost during meta-device init (transformers>=5.x)  说人话就是存的freqs可能丢了，这里检查下
				if self.freqs_cos[0,0] == 0:
						freqs_cos, freqs_sin = precompute_freqs_cis(
								dim=config.head_dim, end=config.max_position_embeddings,
								rope_base=config.rope_theta, rope_scaling=config.rope_scaling
						)
						self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
				position_embeddings = (self.freqs_cos[start_pos: start_pos + seq_len], self.freqs_sin[start_pos: start_pos + seq_len])
				presents= []
				for layer, past_kv in zip(self.layers, past_kv):
						hidden_states, present = layer(
								hidden_states,
								position_embeddings,
								past_kv,
								use_cache,
								attention_mask
						)
						presents.append(present)
				hidden_states = self.norm(hidden_states)
				aux_loss = sum(
						[l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)],
						hidden_states.new_zeros(1).squeeze()
				)
				return hidden_states, presents, aux_loss
```

关于`start_pos`：

没有 cache 时：

```text
past_key_values[0] = None
start_pos = 0
```

有 cache 时，比如已经缓存了 100 个 token：

```text
past_key_values[0][0].shape = [B, 100, num_kv_heads, head_dim]
```

所以：start_pos = 100

## MiniMindForCausalLM

```python
class MiniMindForCausalLM(PreTrainModel, GenerationMixin):
		config_class = MiniMindConfig
		_tied_weights_keys = {"lm_head.weight", "model.embed_tokens.weight"}
		def __init__(self, config: MiniMindConfig = None):
				self.config = config or MiniMindConfig()
				super().__init__(self.config)
				self.model = MiniMindModel(self.config)
				self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
				if self.config.tie_word_embeddings: self.model.embed_tokens.weight = self.lm_head.weight
				self.post_init()  # 初始化钩子，让模型符合 PreTrainedModel 的生命周期

		def forward(self, input_ids, attention_mask=None, past_kv=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
				hidden_states, past_kv, aux_loss = self.model(input_ids, attention_mask, past_kv, use_cache, **kwargs)
				slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
				logits = self.lm_head(hidden_states[:, slice_indices, :])
				loss = None
				if labels is not None:
						x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
						loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ingore_index=-100)
				return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_kv, hidden_states=hidden_states)

		@torch.inference_mode  # 表推理模式，不记录梯度
		def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition=1.0, **kwargs):
			  input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
			  attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
			  past_key_values = kwargs.pop("past_key_values", None)
			  finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
			  if streamer: streamer.put(input_ids.cpu())
			  for _ in range(max_new_tokens):
					  past_len = past_key_values[0][0].shape[1] if past_key_values else 0
					  outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache, **kwargs)
					  attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None  # 在batch_size维度上，为每个batch的attention_mask都新增1个1
					  logits = output.logits[:, -1, :] / temperature
					  if repetition_penalty != 1.0:
							  for i in range(input_ids.shape[0]):
									  seen = torch.unique(input_ids[i])
									  score = logits[i, seen]
									  logits[i, seen] = torch.where(
											  score > 0,
											  score / repetition_penalty,
										    score * repetition_penalty
										)
						if top_k > 0:
								logits[logits < torch.topk(logits, topk)[0][..., -1, None]] = -float('inf')
						if top_p < 1.0:
								sorted_logits, sorted_indices = torch.sort(logits, descending=Ture)
								mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
								mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
								logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
						next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
						if eos_token_id is not None:
								next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
						input_ids = torch.cat([input_ids,next_token], dim=-1)
						past_key_values = outputs.past_key_values if use_cache else None
						if streamer: streamer.put(next_token.cpu())
						if eos_token_id is not None:
								finished |= next_token.squeeze(-1).eq(eos_token_id)  # 更新finished的状态，保证新的EOS也标记为结束
								if finished.all(): break  # 全部都finish了就结束
				if streamer: streamer.end()
				if kwargs.get("return_kv"): return {'generated_ids': input_ids, "past_kv": past_key_values}
				return input_ids
```

`top_p`：累积概率采样
e.g.：

```text
A: 0.40
B: 0.25
C: 0.15
D: 0.08
E: 0.05
F: 0.03
...
```

如果：

```text
top_p = 0.85
```

模型会从最高概率 token 开始累加：

```text
A          0.40
A + B      0.65
A + B + C  0.80
A + B + C + D 0.88
```

累计到超过 `0.85` 后，就只保留：

```text
A, B, C, D
```

其他 token 全部丢掉，不参与采样。

`do_sample`            采样还是贪心

`repetition_penalty`   重复惩罚

`finished`：记录batch里哪些序列已经生成到eos_token_id，[bs, num_return_sequences]

```python
 for i in range(input_ids.shape[0]):
			seen = torch.unique(input_ids[i])
			score = logits[i, seen]
			logits[i, seen] = torch.where(
					score > 0,
					score / repetition_penalty,
			    score * repetition_penalty
			)
```

`seen = torch.unique(input_ids[i])` ：找出当前序列中出现过的token_id并去重

`score = logits[i, seen]` ：找出这些token_id对应的分数

`torch.where(A, B, C)` ：如果满足A条件，则执行B，否则执行C

所以最后一段代码的意思就是指如果score是正数，就除以惩罚系数让它变小；如果是负数就乘系数也会让它变小，总之就是让已出现 token 的 logit 变小，更不容易被选中

`logits[logits < torch.topk(logits, topk)[0][..., -1, None]] = -float('inf')`

这段代码的目的是将logits中小于topk阈值的token对应的值都置为负无穷

`logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')`

<details>
<summary>来自codex的解释</summary>

</details>
