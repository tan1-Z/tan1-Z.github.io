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