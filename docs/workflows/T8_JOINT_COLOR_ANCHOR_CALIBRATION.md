# T8 joint metallicity and CMD anchor calibration

本流程把双星的两类观测放在同一个逐系统模型中，并最终输出
\(p(Z_j\mid \widehat z_{j1},\widehat z_{j2},C_{j1},C_{j2},M_{G,j1},M_{G,j2})\)。
这里 \(Z_j\) 是系统的共同真实金属丰度；每颗星的光谱金属丰度测量仍使用原始
`feh_jcaps_1/2`，对应误差使用 `jc_sigma_m_h_1/2`。

## 光谱项与相对星等偏差

对成员 \(k\in\{1,2\}\)，采用 Student-t 光谱项

\[
\widehat z_{jk}\mid Z_j,M_{G,jk}
\sim t_{\nu}\left(\mu_{jk},
\sqrt{\sigma_{jk}^2+s^2}\right),
\]

\[
\mu_{jk}=Z_j+w_Z(Z_j-Z_{\rm piv})+Z_{\rm off}+b_G(M_{G,jk}),
\qquad Z_{\rm piv}=0\;\mathrm{dex}.
\]

其中 \(s\) 和 \(\nu\) 继续采用已经由双星差值确定的 Student-t 核心参数；两颗星的
差值方差因此含有 \(2s^2\)。\(b_G\) 仍是只由双星差值确定形状的相对偏差；三个模式
统一把整条曲线平移到 \(b_G(4.66)=0\)，不改变它随星等的相对形状。绝对颜色锚点的似然中不放入
\(b_G\)，避免把相对星等趋势与绝对锚点混在一起。

在 `anchors` 和 `parsec_surface` 中，固定 \(w_Z=0\) 时仍拟合常数零点
\(Z_{\rm off}\)。若要拟合
\(w_Z\)，必须提供至少两个具有不同 \(Z_{\rm ref}\) 的颜色锚点；代码会在构造校正器时
拒绝单一参考金属丰度的线性模型。`solar_only` 则同时固定 \(w_Z=0\) 和
\(Z_{\rm off}=0\)。

## CMD 项与绝对锚点

`anchors` 模式仍然不把 PARSEC 颜色当作绝对参考，而是从样本中学习一个低维的内禀
颜色面。为避免在探索阶段引入过多自由度，当前限制为

\[
C_\theta(M_G,Z)=\bm B(M_G)\cdot\bm a
 +Z\,\bm B(M_G)\cdot\bm d,
\]

其中 \(\bm B=(1,x,x^2,x^3)\) 是把 \(M_G\) 线性缩放到 \([-1,1]\) 后的三次基函数。
\(\bm a\) 和 \(\bm d\) 是由数据与锚点共同学习的系数，使用宽的正态先验。这个
低维线性金属丰度形式是当前模型的明确限制，不能代表任意复杂的颜色--金属丰度关系。

颜色项为

\[
C_{jk}\mid M_{G,jk},Z_j
\sim t_4\left(C_\theta(M_{G,jk},Z_j),
\sqrt{\sigma_{C,jk}^2+s_C^2}\right).
\]

在 `anchors` 模式中，绝对信息是太阳颜色锚点

\[
(M_G,Z_{\rm ref},C_{\rm ref},\sigma_C)
=(4.66,0,0.818,0.029\;\mathrm{mag}),
\]

其中 0.029 mag 是该参考测量的不确定度，只用于 `anchors` 模式的软锚点似然。
`solar_only` 不使用这个颜色锚点，也不需要太阳 CMD 窗口。

在 `color_anchor_mode="anchors"` 的软锚点模式中，其似然为

\[
C_{\rm ref}\sim\mathcal N\left(
C_\theta(4.66,0),\,0.029^2\right).
\]

### 简化的 `solar_only` 模式

`solar_only` 只保留原始 JCAPS 金属丰度的 Student-t likelihood，不拟合颜色面，也不加入
CMD 颜色 likelihood。它把 \(Z_{\rm off}\) 固定为 0，因此唯一的抽样全局参数是
`population_weights`；普通总体先验作用于完整的 \(Z\) 网格，不要求网格包含精确的零点。
这是一条人为规定的金属丰度零点约定，不是太阳型恒星样本对 \(Z=0\) 的数据约束。

`solar_only` 不需要 CMD 窗口或窗口误差传播；它仍使用每颗星的 \(M_G\) 来计算
固定的 \(b_G(M_G)\)，但颜色及其误差不进入该模式的 likelihood。

不确定的太阳颜色或任意多组软锚点应使用 `color_anchor_mode="anchors"`；此时上面的
正态锚点似然才会生效，并使用本节开头给出的三次基函数颜色面
\(C_\theta(M_G,Z)=\bm B(M_G)\bm a+Z\,\bm B(M_G)\bm d\)。
这个学习颜色面不属于 `solar_only`。

用户可以传入任意一组 `(absg, z_ref, color_ref, sigma_color)` 锚点。单个锚点只
提供常数绝对校准；多个不同 \(Z_{\rm ref}\) 的锚点才提供随金属丰度变化的校准信息。
锚点直接约束 \(C_\theta\)，并通过每个双星的潜在 \(Z_j\) 与 \(Z_{\rm off},w_Z\)
形成联合似然；它们不直接进入 \(b_G\)。
`color_anchor_mode="parsec_surface"` 才启用旧的固定 PARSEC CMD 似然，此模式没有
学习颜色系数，也不再额外加入锚点因子。

## 输出与限制

对每个系统，三个模式都在完整 \(Z\) 网格上联合边缘化光谱项和总体先验；`solar_only`
没有特殊的锚定系统或 `Z=0` 点质量。`anchors` 模式仍然对颜色项和学习颜色面参数取后验平均，
`parsec_surface` 仍使用固定 PARSEC 颜色面。`solar_only` 的零点只来自
\(b_G(4.66)=0\) 且 \(Z_{\rm off}=0\) 的参数化约定，不来自太阳 CMD 数据。
