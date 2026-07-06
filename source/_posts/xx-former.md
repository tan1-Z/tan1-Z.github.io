---
title: XX-Former范式创新思路
date: 2026-07-06 12:00:00
tags:
  - 科研
categories:
  - 科研
katex: true
---

主线是 **MetaFormer / XX-Former 范式**。它认为很多 Transformer 类模型的成功不一定完全来自 self-attention，而是来自一种通用结构：

$X' = X + TokenMixer(Norm(X))$

$Y = X' + FFN(Norm(X'))$

**先做 token 间的信息混合，再做通道/特征维度的非线性变换。**

TokenMixer主要用于token/patch/point之间的信息交互，可替换为**Attention、CNN、Mamba、FFT、DCT、小波、图结构、超图、低分辨率注意力…**

FFN主要是通道维度的非线性变换 ，可以用Conv-FFN、Gated-FFN、Spectral-FFN、Wavelet-FFN、多尺度 FFN等来替换

Norm和Residual也挺好用，残差爷爷和门控会帮你的^_^

**同时还可以通过多分支来缝多个创新点**

Block Fusion：并联、串联、残差、门控

Stage Fusion：多层特征融合、FPN、U-Net skip connection

1. 局部建模：常用的就是**CNN及其变体（深度可分离卷积、空洞卷积或动态卷积…)**

1. 全局建模：Attention/TransFormer
不过这玩意比较大的问题就是复杂度是$O(N^2)$的，因此也有一套对应的降低Attention代价的方式：把 Query、Key、Value 全部**下采样**到固定低分辨率，在**低分辨率空间计算 attention，再上采样**恢复分辨率，从而降低复杂度

Y = Up(Attention(Down(Q), Down(K), Down(V)))

显然，上采样和下采样也就成了可以创新做文章的地方

**高分辨率上保留细节，低分辨率上计算全局关系（****局部+全局**

1. 线性全局建模：Mamba/SSM
直接用Mamba去替换TokenMixer，不过感觉直接套有点Mamba Out了TuT

1. 频域建模：FFT / DCT / 小波变换
几种常用的讲故事的动机：

| 适合任务 | 频域成分 | 可解释作用 |
| --- | --- | --- |
| 小目标检测、超分、去模糊、去噪 | 高频 | 边缘、纹理、细节、小目标、噪声敏感结构 |
| 分割、补全、重建 | 低频 | 全局轮廓、主体结构、平滑区域 |
| 高效全局建模 | FFT / DCT | 全局频谱交互，低复杂度捕获长程关系 |
| 多尺度细节增强 | 小波 | 同时保留空间位置和频率分量 |

e.g.：高频特征生成器提取频域信息，再进入通道路径和空间路径，形成通道注意力与空间注意力。**频域特征 + 空间特征 + 通道特征的双路径/多路径融合**

1. 多尺度建模：A+B+C
本质是：不同尺度、不同域、不同感受野的特征互补

常见的组合：

  - 小卷积核 + 大卷积核；
  - 普通卷积 + 空洞卷积；
  - 局部分支 + 全局分支；
  - 高频分支 + 低频分支；
  - 空间分支 + 通道分支；
  - CNN + Transformer；
  - CNN + Mamba；
  - Mamba + FFT；
  - CNN + Mamba + FFT；
  - Transformer + Mamba + 频域。
讲故事：

> 单一尺度难以同时处理细节与语义，单一域难以同时捕捉空间结构与频率响应，因此设计多分支模块进行互补建模。

1. 自适应融合
$Y = αA(X) + (1-α)B(X)$

α 可以是可学习参数，也可以由输入动态生成

比add/cat更好讲故事：不同样本、不同区域、不同通道对局部/全局、空间/频域、高频/低频的依赖不同，因此需要自适应权重动态调节

更多分支的时候，只需要保证多个分支的权重总和为1即可

#### 常见套路

1. 替换TokenMixer
原始模块：TokenMixer=Self-Attention

改成：TokenMixer=Mamba / FFT / CNN / Low-Resolution Attention / Wavelet

> 现有 attention 计算复杂度高，因此用 Mamba / FFT / 低分辨率 attention 作为更高效的 token mixer

1. 改造FFN
原始 FFN 通常是：$FFN(X)=W_2σ(W_1X)$

可以改成：FFN(X)=Conv-FFN / Spectral-FFN / Wavelet-FFN / Gated-FFN

> 标准 FFN 主要进行逐 token 的通道变换，缺乏空间/频域/多尺度交互，因此引入卷积、频域滤波或门控机制增强表达

1. 局部+全局双分支
$Y=Fuse(LocalBranch(X),GlobalBranch(X))$

LocalBranch可以用各类卷积、KNN、Local Graph…

GlobalBranch可以用Transformer、Mamba、SSM…

> 局部分支负责纹理、边缘、几何邻域；全局分支负责长程依赖和结构一致性

1. 空间域+频域双分支
$Y=Fuse(Spatial(X),Freq(X))$

> 空间域保留位置结构，频域捕获全局响应和细节频率，两者互补

1. 高频+低频分解
$X=X_{low}+X_{high}$

分解后分别建模：$Y=Fuse(A(X_{low}),B(X_{high}))$

> 低频分量包含全局轮廓，高频分量包含边缘和纹理。对于小目标、去模糊、超分、补全等任务，高频细节尤其关键

1. 降采样计算+上采样恢复
$Y=Up(Core(Down(X)))$

> 在低分辨率空间进行全局建模以降低复杂度，同时通过高分辨率路径或上采样恢复空间细节

1. 多尺度卷积/感受野
$Y=Fuse(Conv_{a×a}(X),Conv_{b×b}(X),DilatedConv(X))$

> 不同目标尺度、不同上下文范围需要不同感受野，因此构造多尺度上下文提取模块
