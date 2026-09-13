---
title: T8a JCAPS Student-t 与 PARSEC CMD 金属丰度后验
date: 2026-09-12
tags: [astronomy, wide-binaries, metallicity, Gaia-XP, PARSEC, Student-t, CMD]
status: 设计文档；待代码实现与运行
---

# T8a：JCAPS Student-t 与 PARSEC CMD 金属丰度后验

## 1. 文档目的与两阶段流程

本文件记录第一阶段：利用宽双星两颗成员的 JCAPS 金属丰度观测、未校正的单星误差，以及 PARSEC 主序 CMD 位置，估计每个双星系统共享的潜在金属丰度后验。该后验作为第二阶段质光关系（mass–luminosity relation, MLR）建模的输入。

流程采用 modular/cut inference：第一阶段先完成金属丰度标尺与每个系统的金属丰度后验；第二阶段只读取这个后验，不让动力学数据反向修改第一阶段全局参数。

以下先给出连续变量的普适定义，再在“当前采用的配置”一节列出离散网格、温度阈值和固定参数。测试样本数和随机种子不属于模型定义。

## 2. 下标、观测量和潜变量

| 符号 | 含义 | 单位或类型 |
|---|---|---|
| \(j\) | 一个宽双星系统的索引 | 整数 |
| \(k\) | 系统内成员索引，通常 \(k\in\{1,2\}\) | 整数 |
| \(q\) | 金属丰度网格点索引 | 整数 |
| \(Z_j\) | 系统 \(j\) 两个成员共享的真实 PARSEC 金属丰度 | dex，定义为 \([M/H]\) |
| \(Z_q\) | 第 \(q\) 个金属丰度网格点 | dex |
| \(\widehat z_{jk}\) | JCAPS 观测金属丰度 | dex |
| \(\sigma_{z,jk}\) | 未校正的 JCAPS 单星误差 | dex |
| \(x_{jk}\) | Gaia \(G\) 波段绝对星等 \(M_G\) | mag |
| \(C_{jk}\) | 消光改正后的 \((G_{\rm BP}-G_{\rm RP})_0\) | mag |
| \(\widehat\sigma_{C,jk}\) | 颜色的 formal 测量误差 | mag |
| \(b_G(x)\) | 随绝对星等变化的相对偏差函数 | dex |
| \(s_z\) | 每颗星共享的额外金属丰度尺度 | dex |
| \(\nu_z\) | Student-\(t\) 自由度 | 无量纲 |
| \(\delta_z\) | JCAPS 相对 PARSEC 金属丰度标尺的共同零点 | dex |
| \(s_C\) | CMD 颜色额外散度 | mag |
| \(p_{\rm pop}(Z)\) | 真实系统金属丰度总体先验密度 | dex\(^{-1}\) |
| \(m_{jk}\) | 是否纳入 CMD likelihood 的固定掩码 | 0 或 1 |
| \(\boldsymbol\psi\) | 第一阶段真正采样的全局参数集合 | 本配置为 \((\delta_z,s_C,\boldsymbol\omega)\) |
| \(P_{jq}(\boldsymbol\psi)\) | 给定一组全局参数时，系统 \(j\) 在 \(Z_q\) 上的归一化后验概率质量 | 无量纲 |
| \(\overline P_{jq}\) | 对第一阶段全局参数后验平均后实际保存的概率质量 | 无量纲 |

实际输入列必须是

\[
\widehat z_{j1}=\texttt{feh\_jcaps\_1},\qquad
\widehat z_{j2}=\texttt{feh\_jcaps\_2},
\]

\[
\sigma_{z,j1}=\texttt{jc\_sigma\_m\_h\_1},\qquad
\sigma_{z,j2}=\texttt{jc\_sigma\_m\_h\_2}.
\]

若输入表缺少这四列，应明确报错。不得替换为经过双星共同金属丰度假设校正的列；那些列已经使用了本模型要检验的共享 \(Z_j\) 信息。

## 3. JCAPS 单星 Student-\(t\) 观测模型

定义 Student-\(t\) 密度

\[
\mathcal T_\nu(y\mid\mu,a)
=
\frac{\Gamma((\nu+1)/2)}
{\Gamma(\nu/2)\sqrt{\nu\pi}\,a}
\left[1+\frac{1}{\nu}\left(\frac{y-\mu}{a}\right)^2\right]^{-(\nu+1)/2},
\]

其中 \(y\) 是被评价的观测值，\(\mu\) 是位置参数，\(a>0\) 是尺度参数，\(\nu>0\) 是自由度，\(\Gamma\) 是 Gamma 函数。对成员 \(k\)，采用

\[
\widehat z_{jk}\mid Z_j,\boldsymbol\psi
\sim
\mathcal T_{\nu_z}\!\left(
Z_j+\delta_z+b_G(x_{jk}),
\sqrt{\sigma_{z,jk}^{2}+s_z^{2}}\right).
\]

在给定 \(Z_j\) 和全局参数 \(\boldsymbol\psi\) 时，两个成员的 JCAPS 观测条件独立：

\[
p(\widehat z_{j1},\widehat z_{j2}\mid Z_j,\boldsymbol\psi)
=\prod_{k=1}^{2}
p(\widehat z_{jk}\mid Z_j,\boldsymbol\psi).
\]

观测位置由共享真实值 \(Z_j\)、共同零点 \(\delta_z\) 和星等相对偏差 \(b_G(x_{jk})\) 组成。报告误差与额外尺度以方差相加；\(\nu_z\) 描述剩余厚尾。这里 \(\delta_z>0\) 的符号约定是：在固定 \(Z_j\)、\(b_G\) 和尺度时，JCAPS 的观测值整体偏高。

当前共同零点的先验为

\[
\delta_z\sim\mathcal N(0,0.3^2),
\]

其中均值和标准差的单位都是 dex。该数值属于当前配置，不是普适模型定义。

函数 \(b_G(x)\) 只表示相对形状，需要参考条件

\[
b_G(x_{\rm ref})=0,
\]

其中 \(x_{\rm ref}\) 是选定的参考绝对星等。该条件不能单独确定 \(\delta_z\)，绝对零点需要 CMD 信息。

## 4. 总体金属丰度先验

连续形式可用截断密度的混合表示：

\[
p_{\rm pop}(Z\mid\boldsymbol\omega)
=
\sum_{r=1}^{R}\omega_r\varphi_r^{\rm trunc}(Z),
\qquad
\omega_r\ge0,\qquad
\sum_{r=1}^{R}\omega_r=1.
\]

这里 \(R\) 是 basis 数量，\(\omega_r\) 是第 \(r\) 个 basis 的权重，\(\varphi_r^{\rm trunc}\) 是在支持区间内归一化的截断密度，\(\boldsymbol\omega\) 是权重向量。若 \(\phi\) 表示标准正态密度，\(\mu_r\) 和 \(\tau_r>0\) 是第 \(r\) 个 basis 的中心和尺度，支持区间为 \([Z_{\min},Z_{\max}]\)，则一个具体的截断高斯 basis 是

\[
\varphi_r^{\rm trunc}(Z)
=
\frac{\phi((Z-\mu_r)/\tau_r)}
{\tau_r\left[
\Phi((Z_{\max}-\mu_r)/\tau_r)
-\Phi((Z_{\min}-\mu_r)/\tau_r)\right]}
\mathbf 1[Z_{\min}\le Z\le Z_{\max}],
\]

其中 \(\Phi\) 是标准正态累积分布函数，\(\mathbf 1[\cdot]\) 是指标函数。也可以替换为其他已归一化的 basis，但必须把其定义和截断归一化写入 metadata。可采用

\[
\boldsymbol\omega\sim\operatorname{Dirichlet}(\boldsymbol\alpha),
\]

其中 \(\boldsymbol\alpha\) 是正的 Dirichlet 超参数向量。总体先验描述样本中潜在真实 \(Z_j\) 的分布，不是把 JCAPS 观测值直接当作真实金属丰度。

## 5. PARSEC CMD likelihood 与固定温度掩码

PARSEC 插值表提供

\[
C_{\rm P}(x,Z),\qquad T_{\rm eff,P}(x,Z),
\]

其中 \(C_{\rm P}\) 是预测颜色，\(T_{\rm eff,P}\) 是有效温度。颜色 formal 误差由 BP/RP 信噪比传播：

\[
\widehat\sigma_{C,jk}
=
\frac{2.5}{\ln 10}
\sqrt{\operatorname{SNR}_{\rm BP,jk}^{-2}
+\operatorname{SNR}_{\rm RP,jk}^{-2}},
\]

其中两个 \(\operatorname{SNR}\) 是 BP 与 RP 通道信噪比，\(\ln\) 是自然对数。

对纳入 CMD 的成员，颜色 likelihood 为

\[
L^{\rm CMD}_{jk}(Z_j\mid\boldsymbol\psi)
=
\mathcal T_{\nu_C}\!\left(
C_{jk}\mid C_{\rm P}(x_{jk},Z_j),
\sqrt{\widehat\sigma_{C,jk}^{2}+s_C^{2}}\right),
\]

其中 \(L^{\rm CMD}_{jk}\) 是 CMD likelihood，\(\nu_C\) 是 CMD Student-\(t\) 自由度，\(s_C\) 是颜色额外散度。当前颜色零点固定为零。

当前 CMD 配置取

\[
s_C\sim\operatorname{HalfNormal}(0.1\ {\rm mag}),
\qquad \nu_C=4.
\]

这里 \(s_C\) 需要拟合，而 \(\nu_C\) 暂时固定。

先在完整网格上构造固定掩码：

\[
m_{jk}
=
\mathbf 1\!\left[
\min_{q\in\mathcal Q}T_{\rm eff,P}(x_{jk},Z_q)
\ge T_{\rm min}\right],
\]

其中 \(\mathbf 1[\cdot]\) 是指标函数，\(\mathcal Q\) 是金属丰度网格索引集合，\(T_{\rm min}\) 是温度阈值。若 \(m_{jk}=1\)，该成员在每个 \(Z_q\) 上都贡献 CMD likelihood；若 \(m_{jk}=0\)，它仍贡献 JCAPS likelihood，但不贡献 CMD likelihood。

## 6. 连续后验与离散实现

定义由 Student-\(t\) 模型得到的光谱 likelihood 为

\[
L^{\rm spec}_{jk}(Z\mid\boldsymbol\psi)
=
\mathcal T_{\nu_z}\!\left(
\widehat z_{jk}\mid Z+\delta_z+b_G(x_{jk}),
\sqrt{\sigma_{z,jk}^{2}+s_z^{2}}\right).
\]

给定全局参数，系统 \(j\) 的连续未归一化后验密度为

\[
\widetilde p_j(Z\mid\boldsymbol\psi)
=
p_{\rm pop}(Z\mid\boldsymbol\omega)
\prod_{k=1}^{K_j}
L^{\rm spec}_{jk}(Z\mid\boldsymbol\psi)
\left[L^{\rm CMD}_{jk}(Z\mid\boldsymbol\psi)\right]^{m_{jk}},
\]

其中 \(K_j\) 是系统成员数，通常为 2。归一化后

\[
p_j(Z\mid\boldsymbol\psi,\mathcal D_j)
=
\frac{\widetilde p_j(Z\mid\boldsymbol\psi)}
{\int_{\mathcal Z}\widetilde p_j(Z'\mid\boldsymbol\psi)\,dZ'}.
\]

这里 \(\mathcal D_j\) 表示系统数据，\(\mathcal Z\) 是金属丰度支持区间，\(Z'\) 是积分哑变量。

在网格上用 quadrature 权重 \(w_q^{(Z)}\) 近似积分：

\[
\widetilde P_{jq}(\boldsymbol\psi)
=
w_q^{(Z)}p_{\rm pop}(Z_q\mid\boldsymbol\omega)
\prod_{k=1}^{K_j}
L^{\rm spec}_{jk}(Z_q\mid\boldsymbol\psi)
\left[L^{\rm CMD}_{jk}(Z_q\mid\boldsymbol\psi)\right]^{m_{jk}},
\]

\[
P_{jq}(\boldsymbol\psi)
=
\frac{\widetilde P_{jq}(\boldsymbol\psi)}
{\sum_{q'\in\mathcal Q}\widetilde P_{jq'}(\boldsymbol\psi)}.
\]

其中 \(q'\) 是求和哑索引，\(P_{jq}(\boldsymbol\psi)\) 是给定全局参数的网格概率质量，不是单星误差；它是中间量，不单独作为第二阶段输入文件保存。

如果从第一阶段后验中选取 \(S_\psi\) 个全局样本用于后验网格平均，则实际保存的 cut-inference 权重可写为

\[
\overline P_{jq}
=
\frac{1}{S_\psi}\sum_{d=1}^{S_\psi}
P_{jq}(\boldsymbol\psi^{(d)}),
\]

其中 \(S_\psi\) 是实际被选来做平均的后验样本数，不一定等于 MCMC 保存的总样本数；\(d\) 是这些入选样本的索引，\(\boldsymbol\psi^{(d)}\) 是相应全局参数样本。这里不用 \(s\) 作样本索引，以免与 \(s_z\) 以及 T8c 的质量尺度 \(s\) 混淆。当前实现默认用给定随机种子无放回抽取至多 128 个后验样本。metadata 必须记录实际 \(S_\psi\)、随机种子，以及入选索引本身或其 digest。第二阶段只读取 \(\overline P_{jq}\)；它不再把 \(Z_j\) 作为 MCMC 中的连续或离散采样变量。

## 7. 当前采用的配置

以下是当前工作流的配置，不是普适公式的限制。

- 输入严格使用 feh_jcaps_1/2 与 jc_sigma_m_h_1/2。
- 金属丰度网格采用 81 个点，支持区间为 \(Z\in[-1.0,0.6]\) dex，积分采用梯形权重。
- CMD 温度阈值为 \(T_{\rm min}=4000\,{\rm K}\)，并要求完整 \(Z_q\) 网格上的最低 PARSEC 温度满足该阈值。
- \(b_G(x)\)、\(s_z\)、\(\nu_z\) 继承采用的 T6d calibration；若作为 plug-in 固定值，必须在 metadata 中记录数值和来源。
- 第一阶段采样的全局参数严格为 \(\boldsymbol\psi=(\delta_z,s_C,\boldsymbol\omega)\)；其中 \(\delta_z\)、\(s_C\) 和 \(\boldsymbol\omega\) 分别按上述先验采样，系统级 \(Z_j\) 在每个全局样本中按 \(Z_q\) 网格求和边缘化。
- 暂不加入旧的质量相关混合状态、\(\chi^2\)-依赖异常概率、灾难性 \(-1.5\) dex 分量或共享 poor-observation offset。

具体地，当前 T6d plug-in 参数为

\[
\begin{aligned}
x\text{ 结点}&=(3.5,\ 6.0,\ 8.5,\ 11.0,\ 13.5)\ {\rm mag},\\
b_G\text{ 结点值}&=(0.1121285,\ 0.0498776,\ 0,\ -0.3103054,\ -0.5332039)\ {\rm dex},\\
s_z&=0.0464650\ {\rm dex},\\
\nu_z&=2.5236494.
\end{aligned}
\]

结点之间的 \(b_G(x)\) 采用线性插值。总体先验当前使用 \(R=6\) 个固定截断高斯基函数：中心 \(\mu_r\) 在 \([-1.0,0.6]\) dex 上等间距，所有基函数的尺度为 \(\tau_r=(0.6-(-1.0))/6\) dex，并在该区间内重新归一化；权重先验为

\[
\boldsymbol\omega\sim\operatorname{Dirichlet}(1,\ldots,1).
\]

当前 PARSEC CMD surface 的构造先筛选

\[
9.3<\log_{10}({\rm age/yr})<10.0,
\qquad 3<G_{\rm mag}<15,
\]

再在 \(3.5\le x=M_G\le13.5\) 的 500 点网格上评价。对每个固定金属丰度，来自所选不同年龄但具有相同 \(G_{\rm mag}\) 的行先分别对颜色和 \(\log T_{\rm eff}\) 取平均；因此年龄在当前 surface 中被折叠，而不是第二阶段的潜变量。颜色—星等曲线采用至多三次的平滑样条，平滑参数为 \(s=0.001\)；\(\log T_{\rm eff}\) 沿星等方向采用线性插值，以避免 4000 K 阈值附近的三次样条过冲，并在建表时指数化为 \(T_{\rm eff}=10^{\log T_{\rm eff}}\)。运行时对已经制表的颜色网格和已指数化的 \(T_{\rm eff}\) 网格分别在 \((M_G,Z)\) 上做双线性插值；温度查询不是继续在 \(\log T_{\rm eff}\) 上插值。年龄范围、源表星等范围、评价网格、重复点聚合规则和两种插值方法都必须保存到 metadata。

在当前基础筛选后的 14,876 个双星系统中，采用 \(T_{\rm min}=4000\ {\rm K}\) 时，29,752 颗成员星中有 12,167 颗进入 CMD likelihood；8,247 个系统至少有一颗成员进入，3,920 个系统的两颗成员都进入。作为覆盖范围的参照，原 \(4730\ {\rm K}\) 阈值对应 6,627 颗成员星、4,901 个至少一颗可用的系统和 1,726 个两颗都可用的系统。这些数字只描述当前输入表经过筛选后的覆盖范围，不是拟合结果，也不能单独说明新增低温恒星与 PARSEC CMD 一致。

## 8. 第二阶段输入与信息边界

第二阶段只需读取网格 \(Z_q\)、实际保存的归一化权重 \(\overline P_{jq}\)、系统/行索引和第一阶段 metadata。动力学边缘化为

\[
\mathcal L_j^{\rm dyn}(\boldsymbol\theta_M)
=
\sum_{q\in\mathcal Q}
\overline P_{jq}\,
L_{j,q}^{\rm dyn}(\boldsymbol\theta_M),
\]

其中 \(\boldsymbol\theta_M\) 是 MLR 参数，\(L_{j,q}^{\rm dyn}\) 是给定 \(Z_q\) 后、已经对速度观测误差卷积并完成 good/outlier 混合的条件动力学 likelihood；左侧 \(\mathcal L_j^{\rm dyn}\) 才是对 \(Z_j\) 后验边缘化后的系统 likelihood。由于 \(\overline P_{jq}\) 已包含总体先验、JCAPS likelihood 和适用的 CMD likelihood，第二阶段不能再次乘这些项或重复加入金属丰度 quadrature 权重。

这种 cut inference 保留单系统 \(Z_j\) 不确定性，但不保留不同系统之间由共同全局参数引起的后验协方差；动力学数据也不会更新 \(\delta_z\)、\(s_C\) 或总体先验权重。

## 9. 解释边界

- \(Z_j\) 是与 PARSEC surface 对接的系统级潜变量，不是两颗星各自独立的真实金属丰度。
- Student-\(t\) 描述当前采用的普通厚尾状态；若要增加灾难性失败状态，应先用原始列进行诊断。
- 4000 K 扩展扩大 CMD 锚定范围，但可能增加低温 PARSEC 颜色模型系统差异；应先检查 CMD residual。
- \(\overline P_{jq}\) 不是把后验中位数当作无误差输入，而是第二阶段对真实金属丰度不确定性进行求和边缘化。
- 4000 K 选择、T6d 固定参数和 cut inference 必须显式写入输出 metadata。
- 当前只通过固定的报告值 \(\sigma_{u,j}\) 在 Rice 模型中传播动力学统计量误差，并通过 \(\overline P_{jq}\) 传播单系统金属丰度不确定性；cut inference 不保留共享 calibration 引起的系统间协方差或动力学反馈，也未传播 \(M_G\)、PARSEC 年龄/模型和固定 T6d plug-in 参数的不确定性。
