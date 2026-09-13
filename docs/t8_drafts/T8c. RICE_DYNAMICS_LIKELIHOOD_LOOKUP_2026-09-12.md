---
title: T8c Rice 动力学 likelihood 与质量尺度 lookup
date: 2026-09-12
tags: [astronomy, wide-binaries, dynamics, Rice, lookup, marginalization]
status: 设计文档；待代码实现与运行
---

# T8c：Rice 积分、动力学 likelihood 与 lookup

本文中 \(\log\) 表示自然对数；质量尺度中的平方根不是对数变换。T8b 中的 \(\log_{10}\) 仅用于定义恒星质量的 dex 表示。

## 1. 作用和符号

本文件记录第二阶段中与 MLR 参数化无关的昂贵积分，以及如何预先查表。T8b 只负责给定绝对星等和金属丰度预测两颗星的质量；T8c 把总质量转换为速度尺度，再与观测速度统计量的测量误差卷积。

| 符号 | 含义 | 单位或类型 |
|---|---|---|
| \(j\) | 双星系统索引 | 整数 |
| \(k\) | 系统成员索引 | 整数 |
| \(q\) | T8a 金属丰度网格索引，\(q=1,\ldots,Q\) | 整数 |
| \(Q\) | 每个系统的金属丰度后验网格点数 | 正整数 |
| \(u_j^{\rm obs}\) | 系统 \(j\) 的观测动力学统计量 \(\Delta v_{\rm tan}\sqrt{r_\perp}\) | km s\(^{-1}\sqrt{\rm AU}\) |
| \(\sigma_{u,j}\) | \(u_j^{\rm obs}\) 的观测误差尺度 | 与 \(u_j^{\rm obs}\) 相同 |
| \(v\) | 积分变量，对应观测统计量空间中的真实幅值 | km s\(^{-1}\sqrt{\rm AU}\) |
| \(\widetilde u\) | 去除质量尺度后的真实归一化统计量 | km s\(^{-1}\sqrt{\rm AU}\) |
| \(s\) | 由总质量决定的固定质量尺度 \(\sqrt{M_{\rm tot}/M_\odot}\) | 无量纲 |
| \(p_c(\widetilde u)\) | 组件 \(c\) 的真实统计量 basis；outlier 严格归一化，当前 good basis 仅近似归一化 | \(\widetilde u^{-1}\) |
| \(c\) | good 或 outlier 组件的标签 | 离散标签 |
| \(L_{c,j}(s)\) | 给定质量尺度的组件动力学概率密度 | (km s\(^{-1}\sqrt{\rm AU}\))\(^{-1}\) |
| \(f_{\rm out}\) | 当前 lookup 约定下的 outlier 有效混合权重 | 0 到 1 |
| \(I_0\) | 第一类零阶修正 Bessel 函数 | 特殊函数 |
| \(\ell_{c,j}\) | 固定上述单位约定后，\(L_{c,j}\) 数值的自然对数 | log-density |

当前质量尺度严格定义为

\[
s_{jq}(\boldsymbol\theta_M)
=
\sqrt{\frac{M_{{\rm tot},jq}(\boldsymbol\theta_M)}{M_\odot}},
\]

其中 \(M_{{\rm tot},jq}\) 是 T8b 在系统 \(j\)、金属丰度节点 \(Z_q\) 下的总质量，\(M_\odot\) 是太阳质量，\(\boldsymbol\theta_M\) 是 MLR 参数。本文不在 lookup 层重新定义 \(s\)；任何改变质量尺度定义的模型都必须重建 lookup。

当输入表没有直接给出 \(u\) 和 \(\sigma_u\) 时，当前构造为

\[
u_j^{\rm obs}
=
4.74\,
\frac{\Delta\mu_j}{\varpi_j}\,
\sqrt{\texttt{sep\_AU}_j},
\qquad
\sigma_{u,j}
=
\frac{u_j^{\rm obs}}{({\rm dpm}/{\rm error})_j},
\]

其中 \(\Delta\mu_j\) 是两成员的 proper-motion 差幅，\(\varpi_j\) 是视差，\(r_{\perp,j}\) 是以 AU 表示的投影分离，而 \(\texttt{sep\_AU}_j\) 是它的数值。4.74 是把 mas yr\(^{-1}\) 与 pc 转为 km s\(^{-1}\) 的换算因子。也就是说，物理统计量是 \(u=\Delta v_{\rm tan}\sqrt{r_\perp}\)，单位为 km s\(^{-1}\sqrt{\rm AU}\)；代码通过乘 \(\sqrt{\texttt{sep\_AU}}\) 实现。\(v\)、\(\widetilde u\)、\(\mu_{\rm out}\) 和 \(\sigma_{\rm out}\) 都采用同一单位约定。若输入直接提供 \(u,\sigma_u\)，则它们必须已采用相同定义。

## 2. Rice 测量模型

若观测的是二维速度矢量的幅值，给定真实幅值 \(v\) 后，观测幅值的 Rice 密度为

\[
R(u\mid v,\sigma)
=
\frac{u}{\sigma^2}
\exp\!\left[-\frac{u^2+v^2}{2\sigma^2}\right]
I_0\!\left(\frac{uv}{\sigma^2}\right),
\qquad u\ge0,
\]

其中 \(u\) 是被评价的观测幅值，\(v\ge0\) 是真实幅值，\(\sigma>0\) 是两个正交分量共同的高斯误差尺度，\(I_0\) 是零阶修正 Bessel 函数。该形式保留了幅值的正定性；在 \(u\) 很小或误差较大时，不能把它无条件替换成一维高斯。

## 3. 真实归一化速度与组件 likelihood

对组件 \(c\)，令 \(p_c(\widetilde u)\) 是真实归一化统计量的 basis。严格生成密度应在其支持区间上积分为 1；当前 outlier 满足该条件，而当前 good basis 与 1 有下文说明的微小差异。质量尺度把归一化统计量转换为真实幅值：

\[
v=s\widetilde u.
\]

因此给定 \(s\) 的组件 likelihood 是

\[
L_{c,j}(s)
=
\int_{\mathcal U_c}
R\!\left(u_j^{\rm obs}\mid s\widetilde u,\sigma_{u,j}\right)
p_c(\widetilde u)\,d\widetilde u,
\]

其中 \(\mathcal U_c\) 是组件 \(c\) 的归一化速度支持区间，\(d\widetilde u\) 是对潜在真实速度的积分微元。这个积分同时完成了真实速度分布与观测误差的卷积。

用 \(v\) 作为积分变量时，\(d\widetilde u=dv/s\)，所以等价形式为

\[
L_{c,j}(s)
=
\int_{s\mathcal U_c}
R(u_j^{\rm obs}\mid v,\sigma_{u,j})
p_c\!\left(\frac{v}{s}\right)
\frac{dv}{s}.
\]

最后的 \(1/s\) 是变量替换的 Jacobian，不能省略。两种形式必须给出相同数值。

当前采用的 good 组件密度为

\[
p_{\rm good}(\widetilde u)
=A\widetilde u
\exp\!\left[
-B\widetilde u^2
-\exp\!\left(\frac{\widetilde u-u_c}{C}\right)
\right]
\mathbf 1[0<\widetilde u\le80],
\]

其中当前固定常数为

\[
A=5.434\times10^{-3},\qquad
B=2.544\times10^{-3},\qquad
u_c=35.67,\qquad
C=3.100.
\]

这些常数共同规定当前动力学母分布的形状和 raw amplitude，不是本轮 MCMC 的自由参数。量纲分别为 \([A]=[\widetilde u]^{-2}\)、\([B]=[\widetilde u]^{-2}\)，而 \(u_c\) 与 \(C\) 和 \(\widetilde u\) 同单位。由于 raw basis 的积分并非恰好为 1，归一化语义见下文。

需要明确：按上述当前常数和 \([0,80]\) 支持区间直接积分，good basis 的总面积为

\[
C_{\rm good}
=
\int_0^{80}
A\widetilde u
\exp\!\left[
-B\widetilde u^2
-\exp\!\left(\frac{\widetilde u-u_c}{C}\right)
\right]d\widetilde u
=0.9978600946.
\]

因此，若要把它解释成严格归一化的生成密度，应使用
\(p_{\rm good}^{\rm norm}=p_{\rm good}^{\rm raw}/C_{\rm good}\)。当前已保存 lookup 沿用 raw basis；故当前的 \(f_{\rm out}\) 是 likelihood 中的有效混合权重，不应直接解释为严格生成模型中的 outlier 概率。若未来把 good basis 归一化，必须重建 lookup，并重新说明 \(f_{\rm out}\) 的含义。

good/outlier 混合模型为

\[
L_j^{\rm mix}(s)
=(1-f_{\rm out})L_{{\rm good},j}(s)
+f_{\rm out}L_{{\rm out},j}(s),
\]

其中 \(L_j^{\rm mix}(s)\) 是给定质量尺度、尚未对金属丰度求和的 good/outlier 混合概率密度，其单位是 \(u^{-1}\)；\(f_{\rm out}\) 是当前 lookup 约定下的有效 outlier 混合权重。若该权重在 MLR MCMC 中是未知参数，只需在已计算的两个组件 likelihood 之外混合；本文件暂不放开 outlier 速度分布的均值和宽度。

## 4. 当前采用的 outlier 分布

当前 outlier 归一化速度在 \([0,U_{\rm out}]\) 上使用截断高斯。令 \(\mu_{\rm out}\) 和 \(\sigma_{\rm out}>0\) 为均值和标准差，\(\phi\) 与 \(\Phi\) 分别为标准正态密度和标准正态累积分布函数，则

\[
p_{\rm out}(\widetilde u)
=
\frac{1}{\sigma_{\rm out}\,C_{\rm out}}
\phi\!\left(\frac{\widetilde u-\mu_{\rm out}}{\sigma_{\rm out}}\right)
\mathbf 1[0\le\widetilde u\le U_{\rm out}],
\]

\[
C_{\rm out}
=
\Phi\!\left(\frac{U_{\rm out}-\mu_{\rm out}}{\sigma_{\rm out}}\right)
-\Phi\!\left(\frac{-\mu_{\rm out}}{\sigma_{\rm out}}\right).
\]

其中 \(\mathbf 1[\cdot]\) 是指标函数，\(C_{\rm out}\) 是截断后的归一化概率。当前配置固定为

\[
\mu_{\rm out}=40,\qquad
\sigma_{\rm out}=13,\qquad
U_{\rm out}=80,
\]

单位与 \(\widetilde u\) 的速度约定一致。此处固定 \(\mu_{\rm out}\) 和 \(\sigma_{\rm out}\)，不在本轮 MLR MCMC 中把它们作为自由参数。

## 5. lookup 的预计算与 MCMC 插值

选择覆盖正式 MLR 质量范围的单调速度尺度网格

\[
\mathcal S=\{s_\ell:\ell=1,\ldots,L_s\},
\]

其中 \(L_s\) 是网格点数量，\(s_\ell\) 是第 \(\ell\) 个速度尺度。在每个系统 \(j\) 和每个 \(s_\ell\) 上，数值积分得到

\[
\ell_{{\rm good},j\ell}
=\log L_{{\rm good},j}(s_\ell),
\qquad
\ell_{{\rm out},j\ell}
=\log L_{{\rm out},j}(s_\ell).
\]

当前实现对每个系统只在观测值附近积分。令

\[
a_j=\max(0,u_j^{\rm obs}-K\sigma_{u,j}),
\qquad
b_j=u_j^{\rm obs}+K\sigma_{u,j},
\]

其中 \(K\) 是局部窗口包含的观测标准差数目。这个窗口把原本 \(v\in[0,\infty)\) 的 Rice 卷积近似为有限区间积分，窗口外的 Rice tail 被丢弃；因此它不是数学上的精确等式，必须通过增大 \(K\) 与节点数、并与更宽窗口或参考积分比较来验证。若标准 Gauss–Legendre 节点和权重为 \((\xi_r,\omega_r)\)，\(-1\le\xi_r\le1\)，则映射后的节点和权重是

\[
v_{jr}=\frac{a_j+b_j}{2}+\frac{b_j-a_j}{2}\xi_r,
\qquad
w_{jr}=\frac{b_j-a_j}{2}\omega_r.
\]

组件积分近似为

\[
L_{c,j}(s)
\approx
\sum_{r=1}^{R}
w_{jr}R(u_j^{\rm obs}\mid v_{jr},\sigma_{u,j})
\frac{1}{s}p_c\!\left(\frac{v_{jr}}{s}\right)
\mathbf 1\!\left(\frac{v_{jr}}{s}\in\mathcal U_c\right),
\]

其中 \(R\) 是局部求积点数，支持区间指标函数防止变量变换后越过组件的物理定义域。当前代码配置为 \(K=10\)、\(R=64\)。这些数值不是模型定义；应通过与更高精度参考积分比较，确认窗口尾部和求积误差能控制 log-likelihood 误差。

MCMC 每次提出新的 \(\boldsymbol\theta_M\) 后，T8b 给出 \(s_{jq}(\boldsymbol\theta_M)\)，只需在 \(\ell_{c,j\ell}\) 上做经过验证的一维插值，再计算

\[
\log L_j^{\rm mix}(s)
=\operatorname{logaddexp}\!\left[
\log(1-f_{\rm out})+\ell_{{\rm good},j}(s),
\log f_{\rm out}+\ell_{{\rm out},j}(s)
\right].
\]

这里 \(\log L_j^{\rm mix}(s)\) 是 \(L_j^{\rm mix}(s)\) 的自然对数，\(\operatorname{logaddexp}(a,b)=\log(e^a+e^b)\) 用于稳定计算两个组件的和；\(\ell_{c,j}(s)\) 是由 lookup 插值得到的组件对数 likelihood。lookup 因而只把 Rice 卷积提前计算，未预先假定 T8b 的具体 MLR。

当前尺度网格为几何网格

\[
0.25\le s_\ell\le2.5,
\qquad L_s=1024,
\]

并在原始 \(s\) 坐标上对保存的对数 likelihood 做线性插值；范围、点数、坐标和插值方法必须写入 lookup metadata。快速测试可暂用 \(L_s=512\)，但正式结果应使用 1024 点并通过直接积分抽查。普适实现也可以选择在 \(\log s\) 上插值，但不能在不重建 lookup 和验证误差的情况下静默切换插值坐标。

若样本包含 \(N\) 个系统，预计算时间复杂度约为 \(O(NL_sR)\)，两张组件表的存储量约为 \(O(2NL_s)\)。MCMC 中每次对 \(Q\) 个金属丰度网格点插值并加权，主要开销约为 \(O(NQ)\)；这里 \(Q\) 是 T8a 的金属丰度网格点数。

## 6. 金属丰度的外层边缘化

对系统 \(j\)，T8a 输出文件中实际保存的是对第一阶段全局后验平均后的 \(\overline P_{jq}\)；给定 MLR 参数，先由 T8b 计算每个 \(Z_q\) 下的总质量和尺度 \(s_{jq}\)，再进行

\[
\mathcal L_j^{\rm dyn}(\boldsymbol\theta_M)
=
\sum_{q\in\mathcal Q}
\overline P_{jq}\,
L_j^{\rm mix}\!\left(s_{jq}(\boldsymbol\theta_M)\right).
\]

其中 \(\mathcal Q=\{1,\ldots,Q\}\) 是金属丰度网格索引集合，\(L_j^{\rm mix}\) 是第 3 节定义的条件混合概率密度；\(\mathcal L_j^{\rm dyn}\) 也具有 \(u^{-1}\) 的单位。代码中的 \(\ell_{c,j}\) 和总 log-likelihood 是在固定 km s\(^{-1}\sqrt{\rm AU}\) 单位约定下对概率密度数值取自然对数，不能解释为脱离单位约定的物理无量纲量。由于 \(\overline P_{jq}\) 已经包含第一阶段的总体先验、JCAPS Student-\(t\) likelihood 和适用的 CMD likelihood，T8c 不应再次乘这些项，也不应再次加入金属丰度 quadrature 权重。

因此当前模型的两层边缘化是：

1. lookup 内部的积分对真实速度 \(\widetilde u\) 边缘化，并通过 Rice 密度考虑 \(u_j^{\rm obs}\) 的测量误差 \(\sigma_{u,j}\)；
2. lookup 外部的 \(Z_q\) 加权和传播 T8a 的真实金属丰度后验不确定性。

这里“边缘化误差”是简写；严格说法是对潜在真实速度和潜在真实金属丰度积分或求和，而误差尺度进入相应的观测模型。

## 7. lookup 复用条件与 metadata

lookup 可在不同 MLR 参数化之间复用，前提是下列内容完全一致：

- 系统行索引及其顺序；
- 每个系统的 \(u_j^{\rm obs}\) 与 \(\sigma_{u,j}\)；
- Rice 密度定义及数值稳定实现；
- good 组件的真实速度分布；
- outlier 支持区间、\(\mu_{\rm out}\)、\(\sigma_{\rm out}\) 与归一化方式；
- 速度尺度网格、积分节点，以及“在原始 \(s\) 坐标上对 log-likelihood 线性插值”的精确约定。

文件 metadata 应保存上述配置、数据摘要或 digest、lookup 质量范围、积分节点数和收敛误差。若行索引或动力学输入 digest 不一致，应停止并重建 lookup，不能静默复用。

质量尺度超出 lookup 网格时不得无提示外推。应在 MCMC 前检查 T8b 先验与 PARSEC 质量范围是否落在 lookup 覆盖区间内；必要时扩大网格并重新验证端点。

计算 Rice 对数密度时，应使用数值稳定的 \(\log I_0\) 或缩放 Bessel 函数实现，避免 \(u_j^{\rm obs}v/\sigma_{u,j}^2\) 较大时发生溢出。

## 8. 旧 lookup 的 outlier 归一化迁移

若旧 lookup 在积分中实际只保留 \([0,80]\)，但 outlier 高斯仍按半无限区间 \([0,\infty)\) 归一化，则旧表可在 \(\mu_{\rm out}=40\)、\(\sigma_{\rm out}=13\) 固定时统一修正。旧归一化常数为

\[
C_{\rm old}=\Phi(40/13)=0.9989542536697612,
\]

新归一化常数为

\[
C_{\rm new}
=
\Phi((80-40)/13)-\Phi(-40/13)
=0.9979085073395224.
\]

由于 likelihood 与归一化常数成反比，旧表的对数值应加上

\[
\ell_{{\rm out},{\rm new}}
=
\ell_{{\rm out},{\rm old}}
+\log\!\left(\frac{C_{\rm old}}{C_{\rm new}}\right)
=
\ell_{{\rm out},{\rm old}}+0.0010473893812422.
\]

等价的 likelihood 乘数为

\[
\frac{C_{\rm old}}{C_{\rm new}}
=1.0010479380850523.
\]

该常数只适用于旧表确实使用上述旧规范、且 \(\mu_{\rm out}\)、\(\sigma_{\rm out}\) 和 \(U_{\rm out}\) 固定的情况。此时无需重新进行 Rice 积分，只需对旧 outlier log-lookup 应用一次上述常数平移。新生成的 lookup 应直接使用 \([0,80]\) 的 \(C_{\rm new}\)，并在 metadata 中写入归一化版本和 migration_applied 标志，防止重复修正。若旧表有数值 floor，应先检查 floor 对该常数修正是否仍保持可接受的误差。

## 9. 当前限制

- 当前是 modular/cut inference：动力学数据不反向更新 T8a 的金属丰度全局参数或 \(\overline P_{jq}\)。
- T8a 的全局参数后验协方差未被系统间完整保留；保存的 \(\overline P_{jq}\) 主要保留单系统的金属丰度不确定性。
- 当前未边缘化绝对星等、颜色、PARSEC 年龄和恒星模型本身的误差；这些不确定性若要加入，需要在 T8b 的质量 surface 或联合模型中显式引入。
- Rice 测量误差与真实速度分布的卷积必须在 lookup 构建时完成；只对 lookup 输出做插值不能替代这一卷积。
- outlier 有效权重可以在两个已查表的动力学分量之间混合，但本轮固定 outlier 均值 40、标准差 13 和支持区间 0–80，不放开其形状参数；在 good basis 未严格归一化的当前约定下，不能把该权重直接当作精确的总体 outlier 概率。
