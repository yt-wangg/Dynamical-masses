---
title: T8b PARSEC 相对修正与二维硬单调 tensor-product spline 质光关系
date: 2026-09-12
tags: [astronomy, wide-binaries, MLR, PARSEC, monotonicity, spline]
status: 设计文档；待代码实现与运行
---

# T8b：PARSEC 相对修正与二维硬单调 MLR

本文中 \(\log_{10}\) 专门表示以 10 为底的质量对数；未带下标的 \(\log\) 出现在 softplus、概率势或数值 likelihood 时，表示自然对数。

## 1. 模型对象与符号

本阶段使用 T8a 输出的系统级金属丰度后验，建模恒星质量如何依赖绝对星等和真实金属丰度。科学模型仍是相对于固定 PARSEC 样条表示 \(g_{\rm P}^{*}\) 的二维修正样条 \(\Delta_*\)，基本恒等式是 \(g=g_{\rm P}^{*}+\Delta_*\)。\(\Theta=\Pi+D\) 只是施加总 \(\log_{10}M\) 硬单调约束的计算坐标，并不表示改回一个与 PARSEC 无关的总质量模型。相对于原始 PARSEC 表格的报告量则是 \(\Delta_{\rm P}=g-g_{\rm P}\)。

| 符号 | 含义 | 单位或类型 |
|---|---|---|
| \(x\) | 绝对星等 \(M_G\) | mag；数值增大代表更暗 |
| \(Z\) | 与 PARSEC 网格一致的真实金属丰度 \([M/H]\) | dex；数值增大代表更富金属 |
| \(g_{\rm P}(x,Z)\) | PARSEC 基线 \(\log_{10}(M_{\rm P}/M_\odot)\) | dex |
| \(g_{\rm P}^{*}(x,Z)\) | 原始 PARSEC 面在当前有限样条基底上的投影 | dex |
| \(\Delta_*(x,Z)\) | 相对于 PARSEC 样条投影的计算修正 | dex |
| \(\Delta_{\rm P}(x,Z)\) | 相对于原始 PARSEC 面的科学修正 | dex |
| \(\epsilon_{\rm P}(x,Z)\) | PARSEC 样条投影误差 \(g_{\rm P}^*-g_{\rm P}\) | dex |
| \(g(x,Z)\) | 修正后的最终 \(\log_{10}(M/M_\odot)\) | dex |
| \(B_i(x)\) | 绝对星等方向的 B-spline basis | 无量纲 |
| \(C_h(Z)\) | 金属丰度方向的 B-spline basis | 无量纲 |
| \(d_x,d_Z\) | 两个方向的 B-spline 多项式次数 | 非负整数 |
| \(\boldsymbol\kappa_x,\boldsymbol\kappa_Z\) | 两个方向的有序结点向量 | mag 与 dex |
| \(\Pi_{ih}\) | PARSEC 在共享 basis 上的控制系数 | dex |
| \(D_{ih}\) | 计算修正 \(\Delta_*\) 的控制系数 | dex |
| \(\Theta_{ih}\) | 最终质量曲面的控制系数 | dex |
| \(K_x,K_Z\) | 两个方向的控制系数数量 | 正整数 |
| \(M_\odot\) | 太阳质量，用作质量单位 | 质量单位 |
| \(\lambda_x,\lambda_Z\) | 两方向控制系数增量的正尺度 | dex |
| \(g_{\rm P,mono}\) | 仅用于诊断的最近单调 PARSEC 投影，不进入 likelihood | dex |
| \(\tau_D,\tau_x,\tau_Z,\tau_{xZ}\) | 修正幅度及三个差分平滑势的正尺度 | dex 或按差分定义的相应单位 |
| \(f_{\rm out}\) | T8c 中 good/outlier 混合的有效比例参数 | 0 到 1 |
| \(\boldsymbol\theta_M\) | MLR 阶段真正采样的参数集合 | 见第 6 节 |

相对于原始 PARSEC 表格的三层科学定义是

\[
g_{\rm P}(x,Z)=\log_{10}\!\left[\frac{M_{\rm P}(x,Z)}{M_\odot}\right],
\]

\[
\Delta_{\rm P}(x,Z)=\Delta\log_{10}M,
\]

\[
\boxed{g(x,Z)=g_{\rm P}(x,Z)+\Delta_{\rm P}(x,Z)}.
\]

这里 \(M_{\rm P}(x,Z)\) 是原始 PARSEC 表给出的预测质量，\(M_\odot\) 是太阳质量。由此，\(\Delta_{\rm P}\) 的科学解释是相对原始 PARSEC 的偏移，而动力学 likelihood 使用最终质量 \(10^{g(x,Z)}M_\odot\)。

## 2. 共享 tensor-product basis

为使控制系数上的单调条件能够保证整个连续曲面单调，PARSEC 基线的有限维投影与计算修正使用同一套 tensor-product B-spline basis。设 \(d_x,d_Z\) 分别是两个方向的 B-spline 次数，\(\boldsymbol\kappa_x\) 与 \(\boldsymbol\kappa_Z\) 是非降结点向量；它们决定定义域、边界重复结点和 \(K_x,K_Z\) 个 basis 的数量。标准 basis 非负且各自构成分割单位：

\[
B_i(x)\ge0,\quad C_h(Z)\ge0,\qquad
\sum_iB_i(x)=1,\quad \sum_hC_h(Z)=1.
\]

结点和次数必须作为模型配置保存，不能只保存控制系数。tensor-product surface 为

\[
g_{\rm P}^{*}(x,Z)
=
\sum_{i=1}^{K_x}\sum_{h=1}^{K_Z}
\Pi_{ih}B_i(x)C_h(Z),
\]

\[
\Delta_*(x,Z)
=
\sum_{i=1}^{K_x}\sum_{h=1}^{K_Z}
D_{ih}B_i(x)C_h(Z).
\]

其中 \(g_{\rm P}^{*}\) 是在选定有限 basis 上对原始 PARSEC surface 的投影，\(\Pi_{ih}\) 是其固定控制系数；\(D_{ih}\) 是相对于该投影拟合的修正系数。定义

\[
\Theta_{ih}=\Pi_{ih}+D_{ih},
\]

则最终曲面满足

\[
g(x,Z)
=
\sum_{i=1}^{K_x}\sum_{h=1}^{K_Z}
\Theta_{ih}B_i(x)C_h(Z).
\]

同时保持 PARSEC-relative 的计算恒等式

\[
\boxed{g(x,Z)=g_{\rm P}^{*}(x,Z)+\Delta_*(x,Z)}.
\]

拟合时应评估 \(g_{\rm P}^{*}\) 与原始 \(g_{\rm P}\) 的逼近误差，并将该误差与动态修正区分开来。不能因为 \(\Pi_{ih}\) 是控制点就把它误解为 PARSEC 表中每个采样点的精确值。

具体定义确定性的投影误差

\[
\epsilon_{\rm P}(x,Z)
=g_{\rm P}^{*}(x,Z)-g_{\rm P}(x,Z).
\]

则真正相对于原始 PARSEC 表面的科学修正为

\[
\boxed{
\Delta_{\rm P}(x,Z)
=g(x,Z)-g_{\rm P}(x,Z)
=\Delta_*(x,Z)+\epsilon_{\rm P}(x,Z)
}.
\]

因此 \(D_{ih}\) 直接表示的是计算修正 \(\Delta_*\)，而不是在原始 PARSEC 每个位置上的精确修正。报告结果时应给出 \(\Delta_{\rm P}\)，并单独检查 \(\epsilon_{\rm P}\) 的 RMS 与最大绝对值。

## 3. 总质量空间中的硬单调约束

在固定 \(Z\) 时，绝对星等数值 \(x\) 增大意味着恒星更暗；物理要求质量不增：

\[
\boxed{\Theta_{i+1,h}\le\Theta_{i,h}}
\qquad
(i=1,\ldots,K_x-1;\ h=1,\ldots,K_Z).
\]

在固定 \(x\) 时，金属丰度增大时质量不减：

\[
\boxed{\Theta_{i,h+1}\ge\Theta_{i,h}}
\qquad
(i=1,\ldots,K_x;\ h=1,\ldots,K_Z-1).
\]

为什么这些不等式能控制导数：对于 \(d_x\) 次 B-spline，导数可写成相邻控制系数差的 B-spline 展开

\[
\frac{\partial g}{\partial x}
=
\sum_{i=1}^{K_x-1}\sum_{h=1}^{K_Z}
\rho_{x,i}(\Theta_{i+1,h}-\Theta_{i,h})
\widetilde B_i(x)C_h(Z),
\qquad \rho_{x,i}>0,
\]

其中 \(\widetilde B_i\) 是由 \(d_x\) 次 basis 降一阶并按结点间距重标定得到的非负 basis，\(\rho_{x,i}\) 是由 \(\boldsymbol\kappa_x\) 和 \(d_x\) 决定的正系数。于是 \(\Theta_{i+1,h}-\Theta_{i,h}\le0\) 时整个导数非正。对 \(Z\) 方向有对应展开

\[
\frac{\partial g}{\partial Z}
=
\sum_{i=1}^{K_x}\sum_{h=1}^{K_Z-1}
\rho_{Z,h}(\Theta_{i,h+1}-\Theta_{i,h})
B_i(x)\widetilde C_h(Z),
\qquad \rho_{Z,h}>0,
\]

其中 \(\widetilde C_h\ge0\) 是降一阶 basis，\(\rho_{Z,h}\) 是由 \(\boldsymbol\kappa_Z\) 和 \(d_Z\) 决定的正系数。因此整个曲面满足

\[
\frac{\partial g}{\partial x}\le0,
\qquad
\frac{\partial g}{\partial Z}\ge0
\]

（在定义域边界采用相应的单侧导数）。在当前采用的 PARSEC 其他输入和定义域内，金属丰度增大时固定 \(x\) 的最终质量不减；这两个条件约束的是 \(\Theta=\Pi+D\)，而不是 \(D\) 单独的系数。

修正量本身可以在任意方向变化。例如 \(D_{i,h+1}>D_{ih}\) 并不是必要条件；只要 PARSEC 与修正相加后的 \(\Theta\) 满足上面的不等式即可。

## 4. 平滑的无约束参数化

直接在 MCMC 中对每个 \(\Theta_{ih}\) 加不等式并在违反时返回负无穷，会产生不连续边界。采用正值增量与区间 sigmoid 构造合法的总控制系数。设 \(a_h\)、\(b_i\) 和 \(r_{ih}\) 是无约束实数，\(\operatorname{softplus}(v)=\log(1+\exp v)\)，\(\operatorname{sigmoid}(v)=(1+\exp(-v))^{-1}\)。令 \(\lambda_x>0,\lambda_Z>0\) 为增量尺度。得到 \(\Theta\) 后，再确定性地计算 \(D=\Theta-\Pi\)，并用收缩势表达“PARSEC 修正通常较小”的偏好；\(D_{ih}\) 不是另一个独立生成层的随机变量。

先生成左上角边界：

\[
\Theta_{1,1}=c_0,
\]

\[
\Theta_{1,h}
=\Theta_{1,h-1}+\lambda_Z\operatorname{softplus}(a_h),
\qquad h=2,\ldots,K_Z,
\]

\[
\Theta_{i,1}
=\Theta_{i-1,1}-\lambda_x\operatorname{softplus}(b_i),
\qquad i=2,\ldots,K_x.
\]

其中 \(c_0\) 是总曲面的基准控制系数，\(\lambda_Z>0\) 和 \(\lambda_x>0\) 是将无约束增量变成 dex 尺度的正尺度。按先构造第一行和第一列、再按递增 \(i+h\) 遍历内部点。若此前已构造的点满足单调性，则

\[
L_{ih}=\Theta_{i,h-1}
\le\Theta_{i-1,h-1}
\le\Theta_{i-1,h}=U_{ih},
\]

这给出内部递归所需的 \(L_{ih}\le U_{ih}\)，并通过对 \(i+h\) 的归纳保证所有新点同时满足两方向不等式。内部点使用

\[
L_{ih}=\Theta_{i,h-1},\qquad
U_{ih}=\Theta_{i-1,h},
\]

\[
\Theta_{ih}
=L_{ih}+(U_{ih}-L_{ih})\operatorname{sigmoid}(r_{ih}),
\qquad i,h\ge2.
\]

由于已构造的边界满足 \(L_{ih}\le U_{ih}\)，内部系数自动满足

\[
\Theta_{i,h-1}\le\Theta_{ih}\le\Theta_{i-1,h}.
\]

这个递归允许混合差分

\[
\Delta_{xZ}\Theta_{ih}
=\Theta_{i+1,h+1}-\Theta_{i+1,h}
-\Theta_{i,h+1}+\Theta_{ih}
\]

为正或为负。这里 \(\Delta_{xZ}\Theta_{ih}\) 是一个控制网格单元的混合差分；不对它的符号加约束，因而没有加入“金属丰度改变时星等斜率必须朝固定方向变化”的额外物理假设。由于 \(g(x,Z)\) 先由同一个标量曲面定义，两个偏导来自同一函数，满足

\[
\frac{\partial}{\partial Z}\left(\frac{\partial g}{\partial x}\right)
=
\frac{\partial}{\partial x}\left(\frac{\partial g}{\partial Z}\right)
\]

在样条足够光滑的区间内；从任一点沿两个坐标方向积分都只是对同一个曲面求值，不会产生路径依赖。严格相等的边界在有限参数下是极限状态，可通过足够小的 softplus 增量逼近。

## 5. 太阳质量锚点

设 \(x_\odot\) 和 \(Z_\odot\) 是太阳参考点的绝对星等与金属丰度，\(g_\odot\) 是太阳参考质量的对数质量（若质量以 \(M_\odot\) 为单位，则 \(g_\odot=0\)）。对总曲面施加

\[
g(x_\odot,Z_\odot)\sim
\mathcal N(g_\odot,\sigma_\odot^2),
\]

其中 \(\mathcal N(\mu,\sigma^2)\) 是均值为 \(\mu\)、方差为 \(\sigma^2\) 的正态分布，\(\sigma_\odot\) 是太阳锚点的对数质量不确定度。

也可以对每个控制点先构造未定零点曲面 \(\widetilde g\)，再做

\[
g(x,Z)=\widetilde g(x,Z)
-\widetilde g(x_\odot,Z_\odot)+g_\odot.
\]

这种确定性中心化保持两方向单调，但会把太阳点固定得过硬；正式分析应根据太阳锚点的不确定度选择概率锚点或中心化加小幅锚点误差。

太阳锚点约束的是最终 \(g\)，不要求 \(\Delta_{\rm P}(x_\odot,Z_\odot)=0\)。因为 PARSEC 太阳点本身可能不完全等于采用的太阳参考质量，\(\Delta_{\rm P}\) 在该点可以有非零值。

## 6. MLR 参数、PARSEC 收缩和显式平滑势

科学修正量是

\[
\Delta_{\rm P}(x,Z)=g(x,Z)-g_{\rm P}(x,Z).
\]

本阶段真正采样的参数定义为

\[
\boldsymbol\theta_M
=
\left(c_0,\{a_h\}_{h=2}^{K_Z},
\{b_i\}_{i=2}^{K_x},
\{r_{ih}\}_{i,h\ge2},
\log\lambda_x,\log\lambda_Z,f_{\rm out}\right).
\]

其中 \(f_{\rm out}\) 是 T8c 的有效混合权重；\(x_\odot,Z_\odot\) 和太阳锚点的误差配置是固定输入，若将太阳锚点中心或尺度也拟合，则应显式加入 \(\boldsymbol\theta_M\)。\(\Pi\)、\(\Theta\)、\(D\)、\(g\)、\(\Delta_*\) 和 \(\Delta_{\rm P}\) 都是由 \(\boldsymbol\theta_M\) 与固定 PARSEC 投影确定性计算出的派生量。

对 raw 参数使用 proper priors，例如

\[
c_0\sim\mathcal N(\mu_0,\tau_0^2),\qquad
\log\lambda_x\sim\mathcal N(\mu_x,\tau_{\lambda x}^2),\qquad
\log\lambda_Z\sim\mathcal N(\mu_Z,\tau_{\lambda Z}^2),
\]

\[
a_h,b_i,r_{ih}\sim\mathcal N(0,\tau_{\rm raw}^2),
\]

其中各参数的具体尺度必须随运行配置保存。由于 \(D\) 是 raw 参数经过递归并减去固定 \(\Pi\) 后的确定性结果，不能再把各个 \(D_{ih}\) 当成互相独立的生成参数。对 PARSEC-relative 修正加入收缩势

\[
\log \pi_{\rm shrink}(D)
=
-\frac{1}{2\tau_D^2}\sum_{i,h}D_{ih}^2+\text{constant},
\]

其中 \(\tau_D>0\) 是计算修正的收缩尺度。显式定义控制网格差分

\[
D^{(2,x)}_{ih}=D_{i+1,h}-2D_{i,h}+D_{i-1,h},
\]

\[
D^{(2,Z)}_{ih}=D_{i,h+1}-2D_{i,h}+D_{i,h-1},
\]

\[
D^{(xZ)}_{ih}
=D_{i+1,h+1}-D_{i+1,h}-D_{i,h+1}+D_{ih}.
\]

若结点间距不等，应将这些差分除以相应的一阶或二阶结点间距。相应的平滑 log potential 为

\[
\log \pi_{\rm smooth}(D)
=
-\frac12\sum_{i,h}
\left(\frac{D^{(2,x)}_{ih}}{\tau_x}\right)^2
-\frac12\sum_{i,h}
\left(\frac{D^{(2,Z)}_{ih}}{\tau_Z}\right)^2
-\frac12\sum_{i,h}
\left(\frac{D^{(xZ)}_{ih}}{\tau_{xZ}}\right)^2
+\text{constant}.
\]

求和只覆盖存在相应邻点的内部索引；这些势鼓励 \(\Delta_*\) 平滑和接近零，但不规定一阶导数或混合差分的符号。

如果原始 PARSEC surface 在金属丰度方向有微小非单调性，普通投影系数 \(\Pi\) 仍应由固定 PARSEC surface 的样条拟合得到，不能静默替换为单调投影。可以另外计算一个仅用于诊断的最近单调投影 \(g_{\rm P,mono}\)，并记录它相对原始 PARSEC 的 RMS 与最大改变量；\(g_{\rm P,mono}\) 不进入 \(\boldsymbol\theta_M\)、不替代 \(g_{\rm P}^{*}\)，也不改变 \(\Theta\) 的硬约束坐标。

## 7. 与动力学 likelihood 的连接

对系统 \(j\) 的成员 \(k\)，令 \(x_{jk}\) 为观测绝对星等，\(Z_q\) 为 T8a 的第 \(q\) 个金属丰度网格点。成员质量为

\[
M_{jkq}(\boldsymbol\theta_M)
=M_\odot\,10^{g(x_{jk},Z_q;\boldsymbol\theta_M)}.
\]

这里 \(M_{jkq}\) 是由 MLR 预测的质量，\(\boldsymbol\theta_M\) 的精确定义见第 6 节。双星总质量是

\[
M_{{\rm tot},jq}
=\sum_{k=1}^{K_j}M_{jkq},
\]

其中 \(K_j\) 是系统成员数，通常为 2。T8c 将这个总质量转换为动力学速度尺度，再与观测 \(u_j\) 的 Rice likelihood 卷积。

令 \(s_{jq}=\sqrt{M_{{\rm tot},jq}/M_\odot}\) 为 lookup 的质量尺度，则系统—金属丰度网格点上的混合动力学 likelihood 为

\[
\mathcal L^{\rm mix}_{jq}
=
(1-f_{\rm out})\,L_{{\rm good},j}(s_{jq})
+f_{\rm out}\,L_{{\rm out},j}(s_{jq}),
\]

其中 \(f_{\rm out}\) 是需要拟合的 outlier 有效混合权重，当前先验为

\[
f_{\rm out}\sim\operatorname{Beta}(3,12).
\]

outlier 分布的均值和标准差保持固定为 40 和 13；它们不是本轮自由参数。

第二阶段完整的金属丰度边缘化为

\[
\mathcal L_j^{\rm MLR}(\boldsymbol\theta_M)
=
\sum_{q\in\mathcal Q}
\overline P_{jq}\,
\mathcal L^{\rm mix}_{jq}(\boldsymbol\theta_M),
\]

其中 \(\overline P_{jq}\) 是 T8a 输出的系统金属丰度后验概率质量，\(\mathcal Q\) 是金属丰度网格索引集合。两个组件的 \(L_{{\rm good},j}\) 和 \(L_{{\rm out},j}\) 均由 T8c 定义并查表。

## 8. T7 原型与 T8b 的关系

T7 使用的

\[
\Delta\log_{10}M(x,Z)=f_0(x)+Zf_Z(x)
\]

是低自由度原型：\(f_0\) 与 \(f_Z\) 各自在少数星等结点上线性插值，且 \(Z\) 方向被固定为线性函数。它不等同于 T3 所建议的低秩或一般 tensor-product spline，也没有把两个方向的硬单调性写成模型构造的一部分。

T8b 保留 PARSEC-relative 解释，但将计算修正 \(\Delta_*(x,Z)\) 提升为二维 tensor-product spline，并把硬约束施加到最终总质量 \(g\) 上；\(\Delta_{\rm P}=\Delta_*+\epsilon_{\rm P}\) 是相对原始 PARSEC 表格的报告量。T7 的 \(f_0,f_Z\) 与 T8b 的 \(D_{ih}\) 因参数化不同，不能逐参数比较；可比较的是最终质量 surface、相对于原始 PARSEC 的 \(\Delta_{\rm P}\)、太阳锚点、动力学预测和留出检验。

## 9. 当前建议配置与限制

以下是首个实现应显式写入 metadata 的配置项，不把测试值写成普适模型定义：

- 使用同一套、已定义结点和阶数的 tensor-product B-spline 表示 \(g_{\rm P}^{*}\) 与 \(\Delta_*\)。
- \(K_x,K_Z,d_x,d_Z,\boldsymbol\kappa_x,\boldsymbol\kappa_Z\) 必须作为配置给出。当前只提出 \(K_x\times K_Z=8\times4\) 作为初始候选；B-spline 次数和完整结点向量尚未决定，必须在首次运行前明确记录，并由数据覆盖、先验敏感性和留出检验核查。第 6 节的参数化共有 \(K_xK_Z+3\) 个采样标量，其中包括 \(\log\lambda_x\)、\(\log\lambda_Z\) 和 \(f_{\rm out}\)；因此 8×4 配置共有 35 个采样标量。这个计数假定所有 \(\tau\) 和太阳锚点配置都是固定超参数。
- 先保持 outlier 分布参数固定为 T8c 的 \(\mu_{\rm out}=40\) 与 \(\sigma_{\rm out}=13\)，不把它们加入本阶段自由参数。
- 质量方向和金属丰度方向只施加一阶硬单调条件；不加入未有物理解释的混合导数符号约束。
- 检查 basis 边界、PARSEC 外推区域、太阳锚点和控制点递归的数值稳定性。

该模型仍依赖 PARSEC 的年龄、颜色、绝对星等和金属丰度定义。硬单调是当前物理假设，不等于证明 PARSEC 或观测系统误差在所有区域都正确；其科学作用是排除与质量排序矛盾的 MLR surface，并通过 \(\Delta_{\rm P}\) 记录相对原始 PARSEC 的数据驱动修正。

当前只通过固定的报告值 \(\sigma_{u,j}\) 在 Rice 模型中传播动力学统计量误差，并通过 \(\overline P_{jq}\) 传播单系统金属丰度不确定性；cut inference 不保留共享 calibration 引起的系统间协方差或动力学反馈，也未传播 \(M_G\)、PARSEC 年龄/模型和固定 T6d plug-in 参数的不确定性。
