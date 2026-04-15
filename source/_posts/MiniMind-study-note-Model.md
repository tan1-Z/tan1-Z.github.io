---
title: MiniMind Model
date: 2026-04-14 10:00:00
tags:
  - Blog
categories:
  - Study Note
  - MiniMindModel
katex: true
---
# Model
这部分主要就是把前面的组件拿来搭积木的事了，整体学习起来也比之前几部分简单些
## MiniMindBlock
```python
class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.self_attn = Attention(
            config.num_attention_heads,
            config.num_key_value_heads,
            config.hidden_size, 
            config.dropout,
            config.flash_attn,
        )

        self.mlp = FeedForward(config) if config.use_moe is False else MoEFeedForward()

        self.layer_id = layer_id

        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_laynorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        
    def forward(self, 
                hidden_states, 
                position_embeddings, 
                past_key_values=None,
                use_cache=False,
                attention_mask=None):

        residual = hidden_states
        hidden_states, past_key_values = self.self_attn(
            self.input_layernorm(hidden_states), 
            position_embeddings, 
            past_key_values,
            use_cache,
            attention_mask,
        )
        hidden_states += residual
        norm_hidden = self.post_attention_laynorm(hidden_states)
        hidden_states = hidden_states + self.mlp(norm_hidden)
        return hidden_states, past_key_values
```
![model structure](/img/minimind.jpg)  
参照这个模型结构图基本上就能理解上面的代码了，常见的Transformer Block结构

## MiniMindModel
```python
class MiniMindModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.drop = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            MiniMindBlock(l, config) for l in range(config.num_hidden_layers)
        ])

        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        freq_cos, freq_sin = precompute_freqs_cis(
            d_model=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            omiga=config.rope_theta,
        )

        self.register_buffer('freq_cos', freq_cos, persistent=False)
        self.register_buffer('freq_sin', freq_sin, persistent=False)

    def forward(self,
                input_ids,
                attention_mask,
                past_key_values,
                use_cache):
        B, T = input_ids.shape
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        hidden_states = self.drop(self.embed_tokens(input_ids))
        
        position_embeddings = (
            self.freq_cos[start_pos: start_pos + T],
            self.freq_sin[start_pos: start_pos + T]
        )

        presents = []

        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value,
                use_cache,
                attention_mask
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        aux_loss = 0.0
        for layer in self.layers:
            if isinstance(layer.mlp, MoEFeedForward):  # 判断每一层的mlp是不是MoEFeedForward，是的话就加上对应的aux_loss
                aux_loss += layer.mlp.aux_loss

        # aux_loss = sum(
        #     layer.mlp.aux_loss
        #     for layer in self.layers
        #     if isinstance(layer.mlp, MOEFeedForward)
        # )
        return hidden_states, presents, aux_loss
```
这里记一下``nn.Embedding()``的``num_embeddings``和``embedding_dim``两个参数的含义：前者是编码了多少token，后者是每个token被编码了多少维

``for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values))``：这一段用zip是为了使每个layer和它对应的past_key_value对应起来，例如：  
```python
self.layers = [layer0, layer1, layer2]
past_key_values = [pkv0, pkv1, pkv2]
```
zip之后就变为了
```python
(layer0, pkv0)
(layer1, pkv1)
(layer2, pkv2)
```

## MiniMindCausalLM
```python
class MiniMindCausalLM(PreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)

        self.model = MiniMindModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # self.embed_tokens.weight = self.lm_head.weight
        self.lm_head.weight = self.model.embed_tokens.weight
        self.OUT = CausalLMOutputWithPast()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                past_key_values=None,
                use_cache=True,
                logits_to_keep=0):
        h, past_kvs, aux_loss = self.model(input_ids,attention_mask, past_key_values, use_cache)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(h[:, slice_indices, :])

        self.OUT.__setitem__('last_hidden_state', h)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', past_kvs)

        return self.OUT
```
``slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep``：这段不太常见的也就slice的作用，在这里等价于[-logits_to_keep: ]，即取最后``logits_to_keep``个位置，注意：``logits_to_keep=0`` 表示全保留    
当然，如果``logits_to_keep``不是整数而是索引的话，就直接拿去用了  
``logits = self.lm_head(h[:, slice_indices, :])``：这里``logits``的维度为[B, K ,vocab_size]，K就是通过slice索引留下来的token数. ``logits_to_keep=1``时，只保留最后一个token预测的分数，适用于自回归生成时；而``logits_to_keep=0``时，全保留，适用于训练场景
