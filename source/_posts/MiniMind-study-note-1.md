---
title: MiniMind study note 1
date: 2026-03-24 00:00:16
tags: 
    - Blog
categories: 
    - Study Note
---
这一部分主要还是对Minimind的每个模块的一些学习，涉及的相关代码在[model_minimind](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py).

# MoE
感觉MoE学起来比较恶心的地方就是它代码里面各种**维度的变换**...  
MoE执行流程：  
1. 每个token用``MoEGate``打分后返回选择的top_k个专家
2. 每个token只输入给它选择的这k个专家
3. 每个专家的输出加权求和，返回输出的结果
## FFN
```python
class FeedForward(nn.Module):
    def __init__(self, 
                 hidden_size: int,
                 intermediate_size: int=None,
                 dropout: float=0.1,
                 hidden_act='gelu'):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = int(hidden_size * 8 / 3)
            intermediate_size = 64 * ((intermediate_size + 64 -1) // 64)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.dropout(
            self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        )
```
这部分还是相对容易理解一些，``gate_proj``和``up_proj``作为双分支融合之后再过一遍  ``down_proj``  
和串行的传统FFN比起来表达能力要强一些

## MoE FFN
![model structure](/source/img/model.png)
```python
class MoEFeedForward(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 n_routed_experts: int,
                 n_shared_experts: int,
                 n_experts_per_tok: int,
                 intermediate_size: int=None,
                 dropout: float=0.1,
                 hidden_act='gelu'):
        super().__init__()
        self.experts = nn.ModuleList(
            FeedForward(hidden_size, intermediate_size, dropout, hidden_act)
            for _ in range(n_routed_experts)
        )

        self.gate = MoEGate(
            n_expert_per_tok=n_experts_per_tok,
            n_routed_experts=n_routed_experts,
            scoring_func='softmax',
            aux_loss_alpha=0.01,
            seq_aux=False,
            norm_topk_prob=True,
            hidden_size=hidden_size
        )
        if n_shared_experts > 0:
            self.shared_experts = nn.ModuleList(
                FeedForward(hidden_size, intermediate_size, dropout, hidden_act)
                for _ in range(n_shared_experts)
            )

        self.n_experts_per_tok = n_experts_per_tok
        self.n_shared_experts = n_shared_experts
```
相关参数解释：  
1. ``n_routed_experts``: 独立处理一些token的专家
2. ``n_shared_experts``: 共享专家，顾名思义，每个token都得走一遍
3. ``n_experts_per_tok``: 其实就是top_k，不知道为啥原码会另起一个这么奇怪的名字. 每个token选择前k个专家  
tips：``MoEGate``会在下一个部分拆解

```python
    def forward(self, x):
        # x [bs, seq_len, hidden_size]
        identity = x
        orig_shape = x.shape
        bs, seq_len, _ = x.shape
        topk_idx, topk_weight, aus_loss = self.gate(x)
        # topk_idx, topk_weight: [bs * seq_len, topk]
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)

        if self.training:
            x = x.repeat_interleave(self.n_experts_per_tok, dim=0)
            # x [(bs * seq_len) * topk, hidden_size]
            y = torch.empty_like(x, dtype=torch.float16)
            for i, expert in enumerate(self.experts):
                mask = (flat_topk_idx == i)
                x_i = x[mask]   # [num_token_i, hidden_size]
                if x_i.shape[0] > 0:
                    y[mask] = expert(x_i).to(y.dtype)

            y = y.view(*topk_weight.shape, -1) # [bs * seq_len, topk, hidden_size]
            y = (y * topk_weight.unsqueeze(-1)).sum(dim=1)
            # y [bs * seq_len, hidden_size]
            y = y.view(*orig_shape) # [bs, seq_len, hidden_size]

        else:
            topk_weight = topk_weight.view(-1, 1) # [bs * seq_len, topk] -> [bs * seq_len * topk, 1]
            y = self.moe_infer(x, flat_topk_idx, topk_weight)
            y = y.view(*orig_shape)

        if self.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        
        self.aux_loss = aus_loss
        return y
```
下面是我在学了一遍代码之后回过头来看还有些疑惑的地方  
``flat_topk_idx = topk_idx.view(-1)`` 为什么要把``topk_idx`` 展平？  
``topk_idx, topk_weight, aus_loss = self.gate(x)``  
 从这里可以看到MoEGate得到的topk_idx的shape是[bs * seq_len, topk]:
- 有bs * seq_len个token
- 每个token会被route到k个专家
- topk_idx[t,k]表示第t个token对应的第k个专家的编号  
但后续在训练时进行了一个repeat的操作：``x = x.repeat_interleave(self.n_experts_per_tok, dim=0)``，导致x的shape变成[bs * seq_len * top_k, hidden_size]     
现在每一行的x对应的都是一个**token-expert pair**，所以要让top_idx的shape跟它对齐，毕竟它是作为route索引在的  
e.g.:
```
x = [token0, token1, token2]
topk_idx =
[
  [1, 3],
  [0, 2],
  [3, 1]
]
```
repeat之后：
```
x =
[
  token0,
  token0,
  token1,
  token1,
  token2,
  token2
]
```
这时候你希望每一行对应的 expert 编号也变成：

``flat_topk_idx =[1, 3, 0, 2, 3, 1]``

这样：

第 0 行 token0 -> expert1  
第 1 行 token0 -> expert3  
第 2 行 token1 -> expert0  
第 3 行 token1 -> expert2  
第 4 行 token2 -> expert3  
第 5 行 token2 -> expert1  
就对齐了
    
```python
    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        # x [bs * seq_len, hidden_size]
        # flat_expert_indices [bs * seq_len * topk]
        # flat_expert_weights [bs * seq_len * topk, 1]
        expert_cache = torch.zeros_like(x)

        idxs = flat_expert_indices.argsort()
        tokens_pre_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.n_experts_per_tok

        for i, end_idx in enumerate(tokens_pre_expert):
            start_idx = 0 if i == 0 else tokens_pre_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx: end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype) # [num_token_i, hidden_size]
            expert_out.mul_(flat_expert_weights[idxs[start_idx: end_idx]])
            # weights_i = flat_expert_weights[idxs[start_idx:end_idx]]
            # expert_out = expert_out * weights_i

            expert_cache.scatter_add_(
                dim=0,
                # [num_token_i] -->  [num_token_i, 1] --> [num_token_i, hidden_size]
                index=exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), 
                # [num_token_i, hidden_size]
                src=expert_out 
            )
        return expert_cache
```