---
title: MiniMind MoE
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
注：``64 * ((intermediate_size + 64 -1) // 64)``为把``intermediate_size``向上取整到最接近的 64 的倍数

## MoE FFN
![model structure](/img/model.png)
```python
class MoEFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.experts = nn.ModuleList(
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        )

        self.gate = MoEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList(
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            )
        self.config = config
```
相关参数解释：  
1. ``n_routed_experts``: 独立处理一些token的专家
2. ``n_shared_experts``: 共享专家，顾名思义，每个token都得走一遍
3. ``n_expert_per_tok``: 其实就是top_k，不知道为啥原码会另起一个这么奇怪的名字. 每个token选择前k个专家  
tips：``MoEGate``会在下一个部分拆解

```python
    def forward(self, x):
        # x [bs, seq_len, hidden_size]
        identity = x
        orig_shape = x.shape
        bs, seq_len, _ = x.shape
        topk_idx, topk_weight, aus_loss = self.gate(x)
        # topk_idx, topk_weight: [bs * seq_len, topk]
        x = x.reshape(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)

        if self.training:
            x = x.repeat_interleave(self.config.n_expert_per_tok, dim=0)
            # x [(bs * seq_len) * topk, hidden_size]
            y = torch.empty_like(x, dtype=x.dtype)
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

        if self.config.n_shared_experts > 0:
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
但后续在训练时进行了一个repeat的操作：``x = x.repeat_interleave(self.config.n_expert_per_tok, dim=0)``，导致x的shape变成[bs * seq_len * top_k, hidden_size]     
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
这里topk_idx的含义是第0个token被路由给专家1,3；第1个token被路由给专家0,2...  
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

``mask = (flat_topk_idx == i) ``  
``x_i = x[mask]   # [num_token_i, hidden_size]``  
这段代码是为了得到repeat后的x中满足flat_topk_idx==i的那些行  
接着用上面的例子，i=3时，可以得到``mask=[False, True, False, False, True, False]``，此时得到的``x_i=[token0, token2]``，token0和token2就是Expert3需要处理的输入  
对应的``x_i``经过相应的Expert处理之后放回y中对应的位置，最后于topk_weight进行加权  

在推理阶段则采用下面的``moe_infer``函数进行处理，该函数的思路如下：     
1. 先把所有路由分支按expert的编号进行排序
2. 把同一个expert负责的token连在一起
3. expert一次性处理自己负责的那批token
4. 把输出按原token位置scatter回去并累加

    
```python
    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        # x [bs * seq_len, hidden_size]
        # flat_expert_indices [bs * seq_len * topk]
        # flat_expert_weights [bs * seq_len * topk, 1]
        expert_cache = torch.zeros_like(x)

        idxs = flat_expert_indices.argsort()
        tokens_pre_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.n_expert_per_tok

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

``idxs = flat_expert_indices.argsort()``是这段代码中比较重要的一部分，``argsort``返回排序后的值对应的位置索引，还是接着用上面这个例子：``flat_topk_idx =[1, 3, 0, 2, 3, 1]``，排序后的expert顺序变为[0, 1, 1, 2, 3, 3]，但``idxs=[2, 0, 5, 3, 1, 4]``，这个idxs表示的含义就是第2个路由分支属于expert 0，第0、5个路由分支属于expert 1...   
``tokens_pre_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)``：经过``bincount``后，得到的结果为[1, 2, 1, 2]表示expert 1~4 分别出现的次数；再经``cumsum``之后，得到[1, 3, 4, 6]，这样就可以得到每个expert对应的区间，比如：expert 1: [0, 1) ; expert 2: [1, 3) ...  
``token_idxs = idxs // self.config.n_expert_per_tok``的作用是把路由分支编号映射回原始的token编号（其实这句话读起来挺绕的，还是举例来说吧：  
假设``n_expert_per_tok=2``，则token_idxs=[1, 0, 2, 1, 0, 2]，即第2条分支来自token 1；第0条分支来自token 0；第5条分支来自token 2...  
```python
        for i, end_idx in enumerate(tokens_pre_expert):
            start_idx = 0 if i == 0 else tokens_pre_expert[i - 1]
            if start_idx == end_idx:
                continue
```  
这段就是去拿每个expert对应的start_idx和end_idx，如果二者相等，说明这个专家没有需要处理的token，故直接跳过  

```python
expert_cache.scatter_add_(
    dim=0,
    # [num_token_i] -->  [num_token_i, 1] --> [num_token_i, hidden_size]
    index=exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), 
    # [num_token_i, hidden_size]
    src=expert_out 
)
```
这个函数还是举例说明吧：  
如果exp_token_idx = [0, 2], 则会把expert_out第0行输出加到expert_cache的第0行；expert_out第2行的输出加到第2行，维度变化只是为了满足函数的使用条件

## MoEGate
```python 
class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.n_expert_per_tok
        self.n_routed_experts = config.n_routed_experts
        
        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux  # True: seq; False: batch

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size

        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim))) # [n_experts, hidden_size]
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, hidden_states):
        # hidden_states: [bs, seq_len, hidden_size]
        bs, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h) # [bs * seq_len, hidden_size]
        logits = F.linear(hidden_states, self.weight, None) # [bs * seq_len, n_experts]

        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')
        
        topk_weight, topk_idx = torch.topk(scores, self.top_k, dim=-1, sorted=False) # [bs * seq_len, top_k]

        if self.top_k > 1 and self.norm_topk_prob:
            d = topk_weight.sum(-1, keepdim=True) + 1e-20
            topk_weight /= d

        # aux loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores # [bs * seq_len, n_experts]
            aux_topk = self.top_k   # [bs, seq_len * topk]
            topk_idx_for_aux_loss = topk_idx.view(bs, -1)
            # [bs * seq_len, topk] -> [bs, seq_len * topk]
          
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bs, seq_len, -1)
                ce = torch.zeros(bs, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(
                    dim=1,
                    index=topk_idx_for_aux_loss,  # [bs, seq_len * aux_topk]
                    src=torch.ones(bs, seq_len * aux_topk, device=hidden_states.device)
                ).div_(seq_len * aux_topk / self.n_routed_experts)

                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha

            else:
                make_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1),
                    self.n_routed_experts,
                )
                ce = make_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (fi * Pi).sum() * self.alpha

        else:
            aux_loss = hidden_states.new_tensor(0.0)
        return topk_idx, topk_weight, aux_loss
```
这段代码里各种维度变换的作用和``scatter_add_``这个艹蛋的函数的使用真的是不得不品的一环（相关的例子也会沿用之间那个  
先给出这些配置项的含义：  
1. ``scoring_func``：打分函数，目前只支持softmax
2. ``aux_loss_alpha``：辅助损失函数的权重
3. ``seq_aux``：为True时对每个样本单独计算它在整个序列上使用了哪些expert；False是整个batch一起计算  
4.``logits``：每个token对expert的分数向量  
如果需要softmax的话，通过softmax把logits转化为概率分布scores，然后找出其topk_weight和idx，在topk>1时，需要对得到的topk_weight再进行归一化  

接下来就是计算aux_loss这个比较关键的环节了  
在``seq_aux=True``时，这里将直接通过一个例子讲解这部分代码在做什么  
假设``bs=2, seq_len=3, top_k=2``， ``topk_idx``如下：
```python
topk_idx =
[
  [1, 3],   # 样本0 token0
  [0, 3],   # 样本0 token1
  [3, 2],   # 样本0 token2
  [1, 1],   # 样本1 token0
  [2, 2],   # 样本1 token1
  [0, 1],   # 样本1 token2
]
```
维度变换后的topk_idx变为了：
```python
[
  [1, 3, 0, 3, 3, 2],   # 样本0 整个序列上的6条路由
  [1, 1, 2, 2, 0, 1]    # 样本1 整个序列上的6条路由
]
```
这样第n行对应的就是样本n在整个序列上使用了哪些expert  
将scores的维度变为[bs, seq_len, n_experts]之后，第0维对应的是第几个样本；第1维对应是这样样本中的第几个token；第2维是这个token对各expert的soft score. 在后续``scores_for_seq_aux.mean(dim=1)``之后就得到了每个样本在整条序列平均后对每个expert的偏好  
接着来讨论``ce``是干啥的，``ce = torch.zeros(bs, self.n_routed_experts, device=hidden_states.device)``，其中ce[b, e]可以理解为第b个样本中，expert e被选中了多少次，根据上面的例子，在有4个expert时，``ce``长下面这样：
```python
ce =
[
  [0, 0, 0, 0],   # 样本0
  [0, 0, 0, 0]    # 样本1
]
```
执行``scatter_add_``这个函数之后，因为 src 对应位置全是 1，所以就是：
- expert1 加 1
- expert3 加 1
- expert0 加 1
- expert3 再加 1
- expert3 再加 1
- expert2 加 1  
最后第0行就变成了[1, 1, 1, 3]，同理第1行就是[1, 3, 2, 0]  
所以其实``ce``的作用就是统计每个样本中，每个expert被选择了多少次的统计表  
后续还有一个归一化操作``.div_(seq_len * aux_topk / self.n_routed_experts)``，这里``seq_len * aux_loss``计算的是应该样本总共有多少次topk路由选择，再除以n_expert之后，得到平均每个expert该被选择多少次  
最后计算aux_loss：``aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha``，本质上是在同时约束硬路由结果(ce)和软路由倾向(score)  

在不使用``seq_aux``的情况下，``ce``还是统计的哪些expert被选择了，后续的计算原理也都一样，只是把整个batch一起算了
