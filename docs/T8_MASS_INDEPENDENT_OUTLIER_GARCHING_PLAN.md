# Garching：默认动力学形状、质量无关 outlier 的全样本计算方案

目标：在同一份 14876 系统金属丰度后验下，测量仅移除 outlier 质量依赖对 MLR 的影响。正常双星动力学形状固定，不做形状采样或交替迭代。

## 已核实的输入与计算环境

2026-09-19 已读取 Garching 上的文件和诊断。

- 主机：`astronode-garching-node03`（Node-03）。
- 项目：`/home/jdli/nexus/collab/Dynamical-masses`。
- Python：`/home/jdli/nexus/miniforge3/envs/dyn/bin/python`；JAX 0.4.18，NumPyro 0.13.0。
- 当前节点可见 192 个 CPU，未检出可用的 GPU 或 sbatch；按普通 CPU 后台任务安排，不直接使用仓库的通用 SLURM 模板。
- 基线目录：`results/hierarchical_metallicity_t8_1_formal/`。
- 金属丰度后验：该目录的 `latent_metallicity_weights_t8.npz`，形状为 14876 × 81。
- 默认动力学表：该目录的 `dynamics_likelihood_lookup_t8.npz`，1024 个质量尺度节点，128 个积分节点，14 sigma 积分窗口。
- 默认 MLR：该目录的 `mlr_mcmc_t8.npz`。4 链，每链 1000 个保留样本；0 次发散，最大 R-hat 1.0080，最小有效样本数约 976。

原始 FITS 使用项目中的 `data/jd_msms_single_bic_1kpc_filtered_cmdcut_cutb_jcaps_err_mhcal_good.fits`。按后验文件的 row_indices 提取并保留顺序，必须得到同样的 14876 个系统。文件名带有 mhcal 不代表使用校正列：金属丰度观测输入是 `feh_jcaps_1/2` 和 `jc_sigma_m_h_1/2`，MLR 实际使用上述保存的隐金属丰度后验。

## 两组模型

令 s_jk=sqrt(M_1(M_G1,Z_k)+M_2(M_G2,Z_k))，q_jk 为保存的金属丰度后验权重。

A 组直接复用默认 formal MLR。其正常双星和 outlier 密度都定义在 w=v/s 上，均带 1/s Jacobian。

B 组拟合

L_j = (1-f) sum_k q_jk L_good,j(s_jk) + f L_out,j，

L_out,j = integral_0^80 Rice(u_obs,j | v, sigma_u,j) TN(v;40,13,[0,80]) dv。

因此 outlier 与质量和金属丰度均无关。40、13 的数值保留，但解释为原始 u 的中心和宽度，单位为 km/s sqrt(AU)；支持区间也固定在原始 u 的 [0,80] 上。Rice 测量误差卷积仍保留。

两组共同固定：

- 正常双星形状 B=0.002544、uc=35.67、C=3.1，并保持单位积分归一化。
- MLR 的 8 × 4 单调张量样条、结点及参数先验。
- tau_D=0.10；tau_x=tau_Z=tau_xZ=0.05 dex。
- 太阳锚点 M_G=4.67、[M/H]=0、log10(M)=0，sigma=0.0043429448 dex。
- f_outlier 仍为自由参数，先验 Beta(3,12)。

## 执行顺序

1. 将 `src/examples/run_mass_independent_outlier.py` 的基线和输出路径改为显式参数，使用 formal 目录；比较图的基线也必须取同一 formal 目录。当前脚本硬编码了 quick 基线，单纯去掉 `--quick` 不会变成全样本。不要改默认模型入口或覆盖已有结果。
2. 读取 formal 后验和 lookup，核对系统顺序、观测输入及固定形状。正常双星 lookup 直接复用。逐系统计算新的 outlier 卷积，保存一维 log likelihood；进入现有拟合器时可在质量网格上广播为常数。
3. 对实际 14876 个系统检查 outlier 卷积与独立自适应积分的一致性；非 floor 项最大对数误差应小于 1e-3，floor 位置需由参考积分确认。检查 outlier 在质量节点上恒定。失败时提高积分精度或修正支持区间，不进入 MLR 拟合。
4. 用全样本做一次单链、100 warmup + 100 draws 的计时试跑，检查有限梯度、内存和实际采样速度。该试跑仅决定资源安排和预热是否需要加长，不混入正式后验。
5. B 组正式跑 4 条独立链，每链 1500 warmup + 2000 个保留样本，target_accept=0.95。用 A 组四条链各自的一个后验样本作为不同起点，独立随机种子 20260920–20260923。用 A 组后验在无约束空间的协方差设置固定稠密 NUTS 度量：0.95 Cov + 0.05 diag(Cov)，其中 f_outlier 转为 logit；只适配步长。该度量用于采样效率，不作为科学先验。
6. CPU 上先按每个进程 8 核限制运行，采用 JAX_PLATFORMS=cpu，BLAS 线程数设为 1；链分进程、分目录保存。在当前负载允许时最多并行 4 链，否则顺序执行。用第 4 步的实测速度估算工时，不根据核心数推算加速比。
7. 合并时保留 chain 维度，检查 rank-normalized R-hat < 1.01、bulk/tail ESS 至少 400、无发散，并检查 f_outlier 和代表性质量值的轨迹。如果只有有效样本数不足，增加保留样本；有发散或链间不一致时先检查采样问题。出现质量 lookup 边界接触时扩展 lookup 并重拟合，不能靠裁剪质量完成结果。

输出根目录建议为 `results/t8_mass_independent_outlier_formal_20260919/`，包含独立链、合并后验、诊断、outlier 积分、模型设置和图表。先写好路径参数与按链保存逻辑，再给出最终启动命令；本方案没有启动新计算。

## 科学输出

- A、B、PARSEC 的 MLR 叠图：[M/H]=-1.0、-0.5、0、+0.3、+0.6，给出 16–84% 区间；标出数据稀疏区域。
- 同一 M_G、[M/H] 下 B/A 的质量中位数相对变化；不把两次独立 MCMC 的 draw 编号当成配对样本。
- M_G=5、7、9、11、13 的质量和后验区间表，分别列 A、B、PARSEC。
- f_outlier 的两组后验；逐源离群概率与原始 u、M_G 的关系，解释质量变化发生在哪些系统。
- 基于相同测量误差与样本选择的观测 u 后验预测图。它用于检查拟合是否能描述观测分布，不单凭训练样本拟合改善宣称模型更真实。

主问题是：quick 中约 4%–7% 的质量下降是否在全样本仍出现；变化是否依赖金属丰度；B 组相对 PARSEC 的偏离还剩多少。

## 局限与金属丰度补验

保存的 formal 金属丰度标定有 3 次发散，最大 R-hat 1.0010、最小有效样本数约 2431。现有 A/B 对照可以回答“条件于这份共同金属丰度后验，改变 outlier 会怎样”，但这些发散尚不能忽略为已解决。

正式定稿前，用同一金属丰度模型补跑 4 链、2000 warmup + 2000 draws、target_accept=0.95，检查发散和全局参数/逐源金属丰度后验的变化。若新旧后验明显变化，A、B 两组必须共同换用新后验重拟合；若变化很小，也需用 B 组的短敏感性复算确认质量曲线的变化小于采样误差，再报告原对照。不得只给 B 组换 metallicity。

这一轮仍保留 PARSEC 相对收缩先验、单调约束、太阳质量锚点和固定动力学形状，因此结果不是完全独立于 PARSEC 的绝对质量标定。这里评估的是 outlier 质量依赖的影响。
