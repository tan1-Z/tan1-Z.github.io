---
title: MiniMind Attention
date: 2026-04-14 10:00:00
tags:
  - Blog
categories:
  - Study Note
  - Attention
katex: true
---

# Attention

主要和普通Attention的不同之处就在于position encoding 和KV cache，所以后续的笔记主要也围绕着两部分展开

## Position Encoding

### 绝对位置编码

```python
def sinusoidal_position_encoding(seq_len, d_model):
    # return tensor dim: [seq_len, d_model]
    position = np.arange(seq_len)[:, np.newaxis] # (seq_len, 1)
    div_term = np.power(10000, 2 * np.arange(d_model // 2) / np.float32(d_model))
    # [10000^(0/d_model), 10000^(2/d_model), 10000^(4/d_model), ...]
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position / div_term)
    pe[:, 1::2] = np.cos(position / div_term)
    return pe # [seq_len, d_model]
```

$$
PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

$$
PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

这两个公式就是绝对位置编码的核心  
``div_term``对应的就是公式里的10000^2i/d_model；  
``pe``就是位置编码矩阵，偶数用sin，奇数cos  

### RoPE 
对于某个位置 $m$ 和某一对维度 $(2i, 2i+1)$，定义旋转角度为：

$$
\theta_{m,i} = m \cdot \omega_i
$$

其中：

$$
\omega_i = \frac{1}{\text{base}^{2i/d}}
$$

然后把这一对维度 $(x_{2i}, x_{2i+1})$

看作二维向量，并施加旋转变换：

$$
R(\theta_{m,i}) =
\begin{bmatrix}
\cos \theta_{m,i} & -\sin \theta_{m,i} \\
\sin \theta_{m,i} & \cos \theta_{m,i}
\end{bmatrix}
$$

$$
\begin{bmatrix}
x'_{2i} \\
x'_{2i+1}
\end{bmatrix}
=
R(\theta_{m,i})
\begin{bmatrix}
x_{2i} \\
x_{2i+1}
\end{bmatrix}
$$

这就是 RoPE 的本质（引自GPT

```python
def precompute_freqs_cis(d_model: int, end: int=(32 * 1024), omiga: int=1e6):
    freqs = 1 / (omiga ** torch.arange(0, d_model, 2)[: d_model // 2].float() / d_model)
    t = torch.arange(end, device=freqs.device).float()
    freqs = torch.outer(t, freqs) # [end * 1, 1 * d_model // 2]
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin

def apply_rotate_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat([-x[...,x.shape[-1] // 2: ], x[..., :x.shape[-1] // 2]], dim=-1)
    q_emb = q * cos.unsqueeze(unsqueeze_dim) + rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    k_emb = k * cos.unsqueeze(unsqueeze_dim) + rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
    return q_emb, k_emb
```

虽然数学公式看着挺吓人，只需要把``freqs``在计算什么弄清楚就行，先通过``freqs = 1 / (omiga ** torch.arange(0, d_model, 2)[: d_model // 2].float() / d_model)``计算得到公式中的W_i，然后把位置向量``t``和``freqs``做外积，最后在进行cos和sin的编码即可

二维旋转公式是：

$$
\begin{aligned}
\begin{bmatrix}
x_1' \\
x_2'
\end{bmatrix}
&=
R(\theta)
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix} \\
&=
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix} \\
&=
\begin{bmatrix}
x_1\cos\theta - x_2\sin\theta \\
x_1\sin\theta + x_2\cos\theta
\end{bmatrix}
\end{aligned}
$$

它可以进一步改写为：

$$
x' = x \cdot \cos\theta + \operatorname{rotate}(x) \cdot \sin\theta
$$

其中：

$$
\operatorname{rotate}(x)=
\begin{bmatrix}
-x_2 \\
x_1
\end{bmatrix}
$$

这就是 RoPE 中

$$
q' = q \cdot \cos\theta + \operatorname{rotate}(q) \cdot \sin\theta
$$

和

$$
k' = k \cdot \cos\theta + \operatorname{rotate}(k) \cdot \sin\theta
$$

的来源  
上面这段原理也引自GPT，注意一下维度的变换即可

## Attention

```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, seq_len, n_kv_head, head_dim = x.shape
    if n_rep == 1:
        return x
    return (x[:, :, :, None, :]
            .expand(bs, seq_len, n_kv_head, n_rep, head_dim)
            .reshape(bs, seq_len, n_kv_head * n_rep, head_dim))
```
一个简单的工具函数，用于复制kv_heads的数量来保证和query一致

```python
class Attention(nn.Module):
    def __init__(self,
                 num_attention_heads: int,
                 num_key_value_heads: Optional[int],
                 hidden_size: int,
                 dropout: float,
                 flash_attn: bool):
        super().__init__()
        self.num_key_value_heads = num_attention_heads if num_key_value_heads is None else num_key_value_heads
        assert num_attention_heads % self.num_key_value_heads == 0

        self.n_local_heads = num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads

        assert hidden_size % num_attention_heads == 0
        self.head_dim = hidden_size // num_attention_heads
        self.q_proj = nn.Linear(hidden_size, self.head_dim * num_attention_heads, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.head_dim * self.n_local_kv_heads, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.head_dim * self.n_local_kv_heads, bias=False)
        self.o_proj = nn.Linear(self.head_dim * num_attention_heads, hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.dropout = dropout

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and flash_attn

    def forward(self,
                x: torch.Tensor, # [bs, seq_len, hidden_size]
                position_embeddings: Tuple[torch.Tensor, torch.Tensor], # rotary pos emb's sin/cos
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, # kv cache
                use_cache=False,
                attention_mask=None):
        bs, seq_len, _ = x.shape

        xq = self.q_proj(x)
        xk = self.k_proj(x)
        xv = self.v_proj(x)

        xq = xq.view(bs, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bs, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bs, seq_len, self.n_local_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        # xq, xk = apply_rotate_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])
        past_len = 0 if past_key_value is None else past_key_value[0].size(1)
        xq, xk = apply_rotate_pos_emb(xq, xk, cos[past_len: past_len + seq_len], sin[past_len: past_len + seq_len])
        

        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)

        past_key_value = (xk, xv) if use_cache else None
        # past_key_value: (k, v), shape [bs, past_len, n_local_kv_heads, head_dim]

        # expand kv heads
        xq = xq.transpose(1, 2) #[bs, n_heads, seq_len, head_dim]
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)

        # Flash Attention
        if self.flash and past_key_value is None:
            dropout_p = self.dropout if self.training else 0.0
            attn_mask = None
            if attention_mask is not None:
                attn_mask = attention_mask.view(bs, 1, 1, -1).expand(bs, self.n_local_heads, seq_len, -1)
                attn_mask = attn_mask.bool()

            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=True
            )

        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # scores  = scores + torch.triu(
            #     torch.full((seq_len, seq_len), float('-inf'), device=scores.device),
            #     diagonal=1
            # ).unsqueeze(0).unsqueeze(0)  # 注意这里是triu不是tril
            kv_seq_len = xk.size(-2)
            causal_mask = torch.triu(
                torch.full((seq_len, kv_seq_len), fill_value=float("-inf"), device=scores.device, dtype=scores.dtype),
                diagonal=kv_seq_len-seq_len + 1,
            )
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)


            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                attention_mask = (1.0 - attention_mask) * -1e9
                scores = scores + attention_mask

            # softmax + attn_drop
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)

            output = scores @ xv
        
        output = output.transpose(1, 2).reshape(bs, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_key_value
```
init部分唯一需要注意的一点就是``q_proj``和``k/v_proj``的out_feats是不同的

```python
past_len = 0 if past_key_value is None else past_key_value[0].size(1)
        xq, xk = apply_rotate_pos_emb(xq, xk, cos[past_len: past_len + seq_len], sin[past_len: past_len + seq_len])
```
第一个需要注意的点就是在计算RoPE时要关注是否需要加上past_len，因为根据下面这个公式：

$$
\theta_{m,i} = m \cdot \omega_i
$$
不同位置的token对应的旋转角是不一样的  

第二个值得剖析的就是关于mask的相关问题  
首先需要明确的一个问题就是attn_scores的shape，根据公式 scores = Q @ K^T，其中``q.shape: [num_q_token, head_dim]``, ``k.shape: [num_k_token, head_dim]``, 故``scores.shape: [num_q_token, num_k_token]``  
基于这个分析，在没有KV cache时，显然attn_scores的shape就是一个方阵；在有KV cache时，因为q_len < k/v_len，attn_scores就不再是一个方阵了，这也引出了causal_mask在计算时的一个小细节
```python
causal_mask = torch.triu(
                torch.full((seq_len, kv_seq_len), fill_value=float("-inf"), device=scores.device, dtype=scores.dtype),
                diagonal=kv_seq_len-seq_len + 1,
            )
```

注意：``diagonal=kv_seq_len-seq_len + 1``，这个参数就是为了保证当前token只能看见自己及之前的token，可以举例如下：  
假设有``[x1, x2, x3, x4, x5]``， 现在KV cache中已经存了前三个，此时输入``[x4, x5]``，则``kv_seq_len=5``, ``seq_len=2``, ``diagonal=4``，得到的mask矩阵如下所示
$$
\text{causal\_mask} =
\begin{bmatrix}
0 & 0 & 0 & 0 & -\infty \\
0 & 0 & 0 & 0 & 0
\end{bmatrix}
$$
$-\infty$表示第4个query看不见第五个k/v