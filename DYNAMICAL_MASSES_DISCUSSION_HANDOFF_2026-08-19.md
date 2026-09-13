# Dynamical Masses 对话交接总结

## 1. 项目与运行环境

### 当前金属丰度输入前提（2026-09-10）

当前项目只使用未校正 XP 金属丰度和未校正误差：

- `feh_jcaps_1`, `feh_jcaps_2`
- `jc_sigma_m_h_1`, `jc_sigma_m_h_2`

`jc_m_h_fit_cal_*` 和 `jc_sigma_m_h_cal_*` 已经使用双星金属丰度相等假设，因此不再作为本项目的输入。本文档中较早的 calibrated 列引用属于历史诊断记录，不代表当前运行约定。

项目目录：

`/Users/ytwang/Library/CloudStorage/OneDrive-Personal/Files/postgraduate/PyProjects/Dyn`

主要子项目：

`bayesian-binary-masses`

Python 脚本和测试应使用 Conda 环境：

```bash
conda run -n dyn python ...
```

主要运行命令：

```bash
conda run -n dyn python -u \
  src/examples/run_on_data_feh_global.py \
  --stage core \
  --heldout-fraction 0.2 \
  --heldout-seed 20260812
```

首次运行一个多小时才出现 MCMC 进度条，主要时间很可能花在 JAX 编译、模型初始化以及大样本似然构建上，而不只是采样本身。

## 2. 已修复的问题

运行结束后，held-out predictive 阶段报错：

```text
TypeError: Dtype >f8 is not a valid JAX array type
```

原因是输入数组为大端字节序 `>f8`，JAX 不能直接转换。

已修改：

`bayesian-binary-masses/src/binary_masses/differencepoly_feh.py`

在 `DifferencePolyFehMLR.set_data` 中将输入统一转换为本机字节序的 `np.float64`。语法检查已通过。

## 3. MCMC 结果分析

结果目录：

`bayesian-binary-masses/results/data_diffpoly2d_anchor_cute_tfeherr_good`

### 数据规模

- 总系统数：15,036
- 训练集：12,029
- held-out：3,007
- 每个模型：4 chains × 2,000 draws = 8,000 posterior draws
- held-out predictive 使用 256 个 posterior draws

### 采样诊断

三个阶段都没有明显采样问题：

| 模型 | Divergence | 最大 R-hat | 最小 bulk ESS | 最大 tree depth |
|---|---:|---:|---:|---:|
| core | 0 | 1.0024 | 1743 | 7 |
| quadratic | 0 | 1.0027 | 1541 | 8 |
| cross | 0 | 1.0031 | 1485 | 8 |

因此 MCMC 数值收敛良好。主要问题是模型物理可解释性和数据系统误差，而不是采样失败。

### Held-out 预测

| 模型 | Held-out lppd | 每系统 lppd |
|---|---:|---:|
| core | -12539.97 | -4.17026 |
| quadratic | -12528.15 | -4.16633 |
| cross | -12524.62 | -4.16515 |

差异：

- quadratic − core：$11.82\pm2.52$
- cross − quadratic：$3.53\pm1.35$
- cross − core：$15.35\pm2.83$

预测上 `cross` 最好，但相对于 `quadratic` 的增益较小，而且几乎全部来自低金属丰度区域：

- $[M/H]<-1$：约 +1.92
- $-1<[M/H]<-0.6$：约 +1.56

这正是当前观测金属丰度最不可靠、PARSEC 网格边界问题最严重的区域，因此不能仅凭 held-out lppd 就断言 cross term 是真实物理效应。

### 关键参数

Quadratic：

$$
a_2=-0.169\pm0.023
$$

显著非零；$b_2$ 的 95% 区间包含 0。

Cross：

$$
c_{xy}=0.085\pm0.028,
\qquad 95\%\ \mathrm{CI}=[0.030,0.141]
$$

但 $b_2$ 仍包含 0。

异常值比例约为：

$$
f_{\rm outlier}\simeq0.196-0.198.
$$

Anchor posterior：

- core：$1.033\pm0.009$
- quadratic：$1.022\pm0.010$
- cross：$1.018\pm0.010$

### 物理诊断

尽管预测改善，quadratic/cross 模型出现较多 MLR 单调性问题：

- core：网格违反比例约 0.00027
- quadratic：约 0.0527
- cross：约 0.0469

金属丰度方向的非递减条件也经常违反，几乎所有 posterior draws 在某些区域存在违反。

相对 PARSEC 的中位质量修正范围很大：

- core：约 $-12.8\%$ 到 $+94.2\%$
- quadratic：约 $-18.1\%$ 到 $+69.8\%$
- cross：约 $-23.4\%$ 到 $+71.0\%$

因此现有多项式参数化有明显外推和非物理形状风险。

## 4. 低金属丰度边界问题

当前 PARSEC metallicity grid 大致只有：

$$
-1.0\le [M/H]\le0.6.
$$

但数据中：

- 2,230/15,036，即 14.83%，观测金属丰度低于 -1
- 57 个高于 0.6
- 671 个系统在 PARSEC 支持区间内的概率低于 5%
- 160 个系统至少一个分量的 $M_G$ 超出约 $[3.5,13.5]$

现有 11 个、围绕观测金属丰度放置的 quadrature nodes 无法处理真正的灾难性错分。例如一颗真实太阳金属丰度恒星若被报告为 $-1.5$，局部积分节点根本不会覆盖真实 $Z\approx0$。

后续建议：

- 扩展 PARSEC metallicity grid；
- 或使用覆盖完整范围的固定全局积分网格；
- 或使用能够同时覆盖观测值附近与总体金属丰度分布的自适应 mixture proposal。

## 5. CMD 图修改

Notebook：

`Model/V5. Process the expanded data.ipynb`

已修改目标 cell：

- 第一个 subplot 的观测恒星范围由 $[-1,-0.8)$ 改为 $[-2,-0.8)$
- 该 subplot 的 PARSEC 曲线仍然使用 $[M/H]=-1.0$ 和 $-0.8$
- 其他 subplot 不变

实现方式是增加：

```python
observed_feh_edges = feh_edges.copy()
observed_feh_edges[0] = -2.0
```

Notebook JSON 和代码语法检查已通过，但旧输出图尚未重新执行生成。

## 6. 数据中发现的观测问题

两个分量都有：

- `bp_rp0_1`, `bp_rp0_2`
- `absg1`, `absg2`
- `jc_m_h_fit_1`, `jc_m_h_fit_2`
- `jc_m_h_fit_cal_1`, `jc_m_h_fit_cal_2`
- `jc_sigma_m_h_1`, `jc_sigma_m_h_2`
- 旧表中还存在 calibrated/inflated uncertainty 列，但它们不属于当前输入
- S/N、flux、alpha 等信息

双星两个分量的金属丰度一致性较差。

原始值：

- $\Delta[M/H]$ 中位数约 0.157 dex
- robust scatter 约 0.423 dex
- 相关系数约 0.441
- 50.7% 的系统两个分量相差超过 0.3 dex
- 34.7% 超过 0.5 dex

校准值：

- 中位差约 0.079 dex
- robust scatter 约 0.373 dex
- 相关系数约 0.439
- 45.5% 超过 0.3 dex
- 31.0% 超过 0.5 dex

当主星 calibrated metallicity 在 $[-2,-0.8]$ 时，共约 3,032 个系统；其中次星有约 16.7% 的金属丰度高于 -0.4。这支持存在明显随机误差和灾难性错分。

颜色 formal uncertainty 很小：

- 主星中位数约 0.0021 mag
- 次星约 0.0085 mag

因此不能只使用 formal error，必须加入额外 CMD scatter 或稳健 likelihood。

## 7. 已解决的代码列选择问题

旧版本 `run_on_data_feh_global.py` 曾经使用：

```python
feh_column = "jc_m_h_fit_1"
feh_sigma_column = "jc_sigma_m_h_cal_1"
```

即：

- 中心值使用未经 calibration 的 `jc_m_h_fit_1`
- 误差却使用 calibrated sigma

当前项目约定已经改为：

```python
feh_column = "jc_m_h_fit_1"
feh_sigma_column = "jc_sigma_m_h_1"
```

原因是 calibrated 列已经使用双星金属丰度相等假设，会和当前 hierarchical calibration likelihood 重复。现有结果中引用 calibrated 列的部分只作为历史结果保留，重新运行必须使用未校正列。

## 8. 建议的新层次模型

核心思想：不要把颜色直接作为 MLR 的第三个自由 predictor。应该令每个双星系统有一个共享的潜在 PARSEC metallicity：

$$
Z_j=[M/H]_{\rm PARSEC}.
$$

两个分量共享 $Z_j$，但各自有不同的观测金属丰度、颜色、星等和测量质量。

### 8.1 金属丰度观测模型

对系统 $j$、分量 $k$：

$$
p(\hat z_{jk}\mid Z_j,q_{jk})
=
(1-\pi_{jk})p_{\rm good}
+
\pi_{jk}p_{\rm bad}.
$$

Good component：

$$
p_{\rm good}
=
t_{\nu_z}\!\left[
\hat z_{jk};
a_0+a_1Z_j+b(q_{jk}),
\sqrt{s_{z,jk}^2+s_{\rm floor}^2}
\right].
$$

其中：

- $\hat z_{jk}$：观测金属丰度；
- $Z_j$：真实/PARSEC metallicity；
- $b(q)$：依赖颜色、星等、S/N 等的观测管线系统差；
- $s_{\rm floor}$：额外误差地板；
- Student-$t$ 用来稳健处理普通重尾误差。

Bad component：

$$
p_{\rm bad}(\hat z_{jk}\mid q_{jk})
$$

表示灾难性失败后，管线通常输出到什么位置。

### 8.2 为什么 bad density 可用观测总体分布

灾难性错误不意味着管线在允许范围内均匀乱猜，而通常意味着结果与真实 $Z$ 失去关系、却仍落在该管线常见输出区域，例如回归到太阳金属丰度或训练集中心。

因此：

$$
\hat z\perp Z\mid B=1,q,
$$

但

$$
p_{\rm bad}(\hat z\mid q)
$$

通常不是 uniform。

更成熟的选择是从低质量、重复观测不一致、双星分量严重不一致或外部标定失败样本中学习一个平滑 density，例如少量高斯混合。最好进行交叉拟合，避免用同一数据同时定义 bad density 和评价预测性能。

需要区分：

- $\pi_{jk}$：这次测量发生灾难性失败的概率；
- $p_{\rm bad}(\hat z\mid q)$：失败后通常输出到哪里。

例如：

$$
\operatorname{logit}\pi_{jk}
=
\gamma_0+\gamma_1\log{\rm S/N}_{jk}
+\gamma_2C_{jk}+\gamma_3M_{G,jk}.
$$

### 8.3 为什么双星差值约束 $b(C,M_G)$

因为两个分量共享 $Z_j$：

$$
\hat z_{j1}-\hat z_{j2}
=
b(q_{j1})-b(q_{j2})
+\epsilon_{j1}-\epsilon_{j2}.
$$

共享的 $Z_j$ 和整体 zero point 会抵消，因此双星内部差值可以约束颜色、星等和 S/N 相关系统差的相对形状。

但双星差值不能确定绝对 zero point；$a_0,a_1$ 仍需要外部高质量 metallicity anchors 或强先验。

## 9. 颜色 likelihood

第一版建议不引入自由的 $\delta C$，而使用：

$$
\hat C_{jk}
\sim
t_{\nu_C}\!\left[
C_{\rm PARSEC}(M_{G,jk},Z_j),
\sqrt{\sigma_{C,jk}^2+s_C^2}
\right].
$$

其中：

- $\hat C_{jk}=(G_{\rm BP}-G_{\rm RP})_{0,jk}$
- $C_{\rm PARSEC}$：PARSEC 给定 $M_G,Z$ 的颜色
- $s_C$：额外 CMD scatter
- Student-$t$：处理未解析多星、异常消光、活动性和离群点

颜色的作用是帮助约束潜在 $Z_j$，而不是直接作为 MLR 的自由 predictor。

## 10. 当前关于 $\delta C$ 的结论

用户已经为观测 $M/H$ 与 PARSEC metallicity 之间的系统差建模，因此第一版最好设：

$$
\delta C(M_G,Z)=0.
$$

理论上两者描述不同误差。

金属丰度系统差：

$$
\hat z_{\rm obs}
=
a_0+a_1Z_{\rm PARSEC}+b(q)+\epsilon_z.
$$

颜色 discrepancy：

$$
\hat C
=
C_{\rm PARSEC}(M_G,Z_{\rm PARSEC})
+\delta C(M_G,Z_{\rm PARSEC})
+\epsilon_C.
$$

但两者高度简并：

$$
C_{\rm PARSEC}(M_G,Z+\Delta Z)
\approx
C_{\rm PARSEC}(M_G,Z)
+
\frac{\partial C}{\partial Z}\Delta Z.
$$

所以颜色偏移既能被解释为 metallicity offset，也能被解释为 $\delta C$。若两者同时自由，数据难以识别，甚至可能把 PARSEC 颜色误差错误传播成 MLR 的金属丰度依赖。

推荐步骤：

1. 用双星一致性、重复观测和外部标定约束 metallicity measurement model；
2. 固定或传播其不确定度；
3. 设置 $\delta C=0$，只加入 $s_C$ 和稳健颜色 likelihood；
4. 检查后验颜色残差：

   $$
   r_C=\hat C-C_{\rm PARSEC}(M_G,Z);
   $$

5. 只有残差随 $M_G$ 或 $Z$ 呈稳定结构时，才加入低维、强收缩的 $\delta C$。

若需要 $\delta C$，先从简单的一维函数开始，例如：

$$
\delta C(M_G)=\beta_0+\beta_1(M_G-M_{G,0}),
$$

不要直接使用自由的二维 $\delta C(M_G,Z)$。

## 11. 质量模型

建议写成相对 PARSEC 的 log-mass correction：

$$
\log M_{jk}
=
\log M_{\rm PARSEC}(M_{G,jk},Z_j)
+
\Delta_M(M_{G,jk},Z_j).
$$

不建议继续无限增加全局高阶多项式项。推荐低秩 tensor-product spline：

$$
\Delta_M(M_G,Z)
=
f_0(M_G)+\lambda g_1(M_G)h_1(Z),
$$

并加入：

- 平滑先验；
- 对 $\lambda$ 的收缩先验；
- MLR 对 $M_G$ 的单调性约束；
- 必要时对 metallicity 方向施加物理先验。

第一版 rank-1 足够，只有 held-out predictive 明确支持时才增加 rank。

## 12. 完整单系统似然

在暂不使用 $\delta C$ 时：

$$
\begin{aligned}
\mathcal L_j
={}&
\int dZ_j\,
p_{\rm pop}(Z_j)
\prod_{k=1}^{2}
p(\hat z_{jk}\mid Z_j,q_{jk})\\
&\times
\prod_{k=1}^{2}
p\!\left(
\hat C_{jk}\mid
C_{\rm PARSEC}(M_{G,jk},Z_j),s_C
\right)\\
&\times
p\!\left[
\hat u_j\mid
M_1(M_{G,j1},Z_j)+M_2(M_{G,j2},Z_j)
\right].
\end{aligned}
$$

其中：

- $\hat z_{jk}$：观测 metallicity
- $\hat C_{jk}$：观测 dereddened color
- $M_{G,jk}$：观测绝对星等，第一版可视为误差较小
- $Z_j$：潜在真实/PARSEC metallicity
- $M_{jk}$：模型预测的潜在恒星质量
- $\hat u_j$：轨道/统计动力学观测量

离散全局金属丰度网格形式：

$$
\mathcal L_j
\approx
\sum_q \Delta Z_q\,
p_{\rm pop}(Z_q)
L^z_{j1q}L^z_{j2q}
L^C_{j1q}L^C_{j2q}
L^u_{jq}.
$$

程序结构可写成：

```python
log_weight_jq = (
    log_grid_weight_q
    + log_population_jq
    + log_metallicity_1_jq
    + log_metallicity_2_jq
    + log_cmd_1_jq
    + log_cmd_2_jq
)

log_likelihood_j = logsumexp(
    log_weight_jq + log_dynamics_jq
)
```

注意：若已有的 `feh_weights` 已包含光谱 metallicity likelihood，就不能再额外加入一次 `log_metallicity`，否则会重复计数。

## 13. 推荐的分阶段实现

### 阶段 A：金属丰度观测模型

利用：

- 双星共享 metallicity；
- 两分量差值；
- S/N、颜色、星等；
- 重复观测和外部标定；

拟合：

- $a_0,a_1$
- $b(C,M_G,\mathrm{S/N})$
- $s_{\rm floor}$
- $\pi(q)$
- $p_{\rm bad}$

### 阶段 B：加入 CMD

在 $\delta C=0$ 下，用颜色 likelihood 更新每个系统的：

$$
p(Z_j\mid\hat z_{j1},\hat z_{j2},\hat C_{j1},\hat C_{j2}).
$$

检查颜色 residual 和 posterior predictive。

### 阶段 C：动力学 MLR correction

将阶段 B 的 $p(Z_j)$ 固定或传播其不确定度，拟合低秩、单调的 $\Delta_M(M_G,Z)$。

### 阶段 D：敏感性分析

比较：

- 不使用颜色；
- 使用颜色但 $\delta C=0$；
- 加简单 $\delta C(M_G)$；
- 不同 bad density；
- 不同 metallicity grid；
- 不同 held-out seeds；
- 是否包含 $[M/H]<-1$ 的边界样本。

## 14. 下一次对话最值得先处理的事项

1. 阅读现有 metallicity quadrature 和 `feh_weights` 的具体实现，判断当前 likelihood 是否已经包含观测金属丰度项。
2. 设计最小可识别版本：
   - 共享 $Z_j$
   - metallicity good/bad mixture
   - $\delta C=0$
   - 一个额外 $s_C$
   - 固定全局 metallicity grid
3. 先做阶段 A 的双星 metallicity calibration，而不是直接把全部新参数塞进现有 MCMC。
4. 用模拟数据做 recovery test，确认：
   - 能识别 catastrophic outliers；
   - 不会把颜色 discrepancy 误认为 metallicity correction；
   - 能恢复已知的 MLR metallicity dependence。
