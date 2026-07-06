---
title: Partial思想
date: 2026-07-06 12:10:00
tags:
  - 科研
categories:
  - 科研
katex: true
---

## Ghost Net

通过部分通道处理降低计算量、参数量：一部分恒等处理，一部分做需要的操作

可用于主干/特征金字塔

目的：减少冗余的特征图（CVPR 2020 GhostNet）

可以通过部分处理的思路加入各式卷积：（解决单一感受野等问题…

- 重参数卷积

- 特性卷积

- 多尺度卷积

- 大感受野卷积

- 动态卷积

- 可变性卷积

```python
class GhostNet(nn.Moudle):
	def __init__(self, inc, outc, dw_conv=5):
		super.__init__()
		c_ = outc // 2
		
		self.primary_conv = nn.Conv2d(inc, c_, 3)
		self.cheap_conv = XXXconv(...)  // 自定义特殊卷积操作
	
	def forward(self,x):
		x = self.primary_conv(x)
		x = torch.cat([x, self.cheap_conv(x)],1)
		return x
```

## Faster Net

仅对1/4的通道处理，其余恒等

```python
class FasterNet(nn.Moudle):
	def __init__(self, inc, outc, n_div=4):
		super.__init_()
		self.partial_c = inc // n_div
		self.identity_c = inc - self.partial_c
		self.partial_conv = XXXconv(...)
		self.conv_adjust = nn.Conv2d(inc, outc, 1) if inc != outc else nn.Identity()
		
	def forward(self,x):
		x1, x2 = torch.split(x, (self.partial_c, self.identity_c),1)
		x1 = self.partial_conv(x1)
		x = torch.cat([x1,x2],1)
		return self.conv_adjust(x)
```

## SHViT

prob：多头注意力机制后期头相似度较高，冗余，移除大多数头不会显著影响性能

对部分通道做Single Head Attention

一个通用思路：

```python
x1,x2,x3 = torch.split(x)
x1 = Global(x1)
x2 = Local(x2)
x = torch.cat([x1,x2,x3])
```

Global：Transformer   Mamba  傅里叶卷积等具有全局感受野的模块

Local：各种卷积…或者一些其他模块
