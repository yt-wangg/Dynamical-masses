# Dynamical Masses 工作交接记录

这份记录用于在新对话中延续全局金属丰度质量模型的诊断、改进和服务器运行讨论。

## 研究目标

原始结果位于：

```text
results/data_diffpoly2d_anchor_cute_tfeherr
```

模型使用真实测量的金属丰度误差，主要配置包括：

- `use_feh_uncertainty=True`
- `feh_sigma_column="jc_sigma_m_h_cal_1"`
- `feh_quadrature_nodes=11`
- `feh_min=-1`
- `feh_max=0.6`
- `feh_model="quadratic2d"`
- `anchor_enabled=True`
- 锚点：`M(AbsG=4.67, FeH=0)=1.0 ± 0.01`
- `cross_mode="free"`
- `quad_mode="full"`
- 4 chains
- 1200 warmup
- 2000 posterior samples
- dense mass matrix

## 原始结果诊断

使用以下 notebook 检查：

```text
src/examples/check_feh_global_mcmc_diagnostics.ipynb
```

得到的结论：

- 原始 TXT 样本的 R-hat 和 ESS 表面上很好。
- 旧输出丢失了 NUTS 的 `diverging`、`energy`、`num_steps`、`accept_prob` 等信息，因此无法完整判断 divergence、BFMI 和树深饱和。
- 科学结果异常的主要嫌疑不是普通意义上的链不收敛，而是二维二次模型中的交叉项 `c_xy`。
- `c_xy × AbsG × FeH` 会随金属丰度旋转或倾斜整条质量—绝对星等关系，容易与线性项、二次项及金属丰度误差模型产生退化。
- 单个锚点只约束一个位置，不能阻止曲线在锚点周围旋转。
- 因此需要分阶段判断额外复杂度是否真的改善样本外预测。

## 已完成的代码修改

### 完整保存 MCMC 诊断

修改了：

```text
src/binary_masses/differencepoly_feh.py
```

增加了：

- `collect_diagnostics`
- 保存 `diverging`
- 保存 `energy`
- 保存 `potential_energy`
- 保存 `num_steps`
- 保存 `accept_prob`
- 计算 held-out 数据在每个 posterior draw 下的 log likelihood

### 三阶段模型比较

修改了：

```text
src/examples/run_on_data_feh_global.py
```

新增三个阶段：

- `core`：只拟合基本结构，关闭纯二次项和 `c_xy`。
- `quadratic`：加入纯二次项，但仍关闭 `c_xy`。
- `cross`：允许完整二次模型和 `c_xy`。

三个阶段必须使用完全相同的 held-out split，然后比较 held-out predictive log likelihood（lppd）。

运行命令：

```bash
python -u src/examples/run_on_data_feh_global.py \
  --stage core \
  --heldout-fraction 0.2 \
  --heldout-seed 20260812
```

之后分别把 `core` 换成 `quadratic` 和 `cross`。三个阶段完成后运行：

```bash
python -u src/examples/run_on_data_feh_global.py \
  --compare-only \
  --heldout-fraction 0.2 \
  --heldout-seed 20260812
```

脚本现在还会：

- 保存 grouped posterior 与 sample statistics 到压缩 NPZ。
- 保存固定的训练/held-out indices。
- 保存 held-out pointwise log likelihood 和 lppd。
- 检查三个阶段是否使用完全相同的 held-out 样本。
- 报告成对 `Δlppd ± SE`。
- 将残差图改成百分比 `(M_fit - M_iso) / M_iso × 100%`。
- 主质量图继续使用对数轴。

### 输入数据和输出目录

当前输入文件：

```text
data/jd_msms_single_bic_1kpc_filtered_cmdcut_cutb_jcaps_err_mhcal_good.fits
```

金属丰度及误差列：

```text
jc_m_h_fit_1
jc_sigma_m_h_cal_1
```

新输出目录：

```text
results/data_diffpoly2d_anchor_cute_tfeherr_good
```

数据共 15,036 行。使用 20% held-out 后，训练集约有 12,029 个系统。

### Notebook 与说明文档

诊断 notebook 已修改为自动识别新 grouped NPZ 或旧 TXT，并检查：

- divergence
- R-hat 和 ESS
- BFMI
- acceptance probability
- NUTS tree-depth saturation

工作流说明位于：

```text
src/examples/FEH_GLOBAL_WORKFLOW.md
```

其中记录了三阶段命令、held-out lppd 的解释、FeH 支持区间问题，以及硬单调 B/I-spline 或 monotone-lattice 参数化的后续思路。

## 金属丰度边界问题

在完整数据中：

- 约 14.83% 的观测金属丰度低于 `FeH=-1`。
- 约 0.38% 高于 `FeH=0.6`。
- 使用 FeH uncertainty 时，这些行不会被简单删除，而是通过误差积分计算其落入模型支持区间的概率。
- 约 4.46% 的恒星在 `[-1, 0.6]` 内的高斯概率低于 5%。

需要留意边缘目标是否异常影响结果。后续可以扩大 FeH 支持区间或对低支持概率样本做敏感性分析，但不应不加检验地依赖多项式外推。

## 已完成的本地验证

以下检查已通过：

- 修改文件的 Python 编译。
- notebook JSON 和所有 code cells 编译。
- `git diff --check`。
- CLI `--help`。
- 小规模 NumPyro smoke test。
- held-out log likelihood 的形状和有限性检查。
- grouped posterior/sample statistics 输出检查。
- 阶段比较 helper 测试。
- 百分比残差图渲染检查。

`dyn` 环境中没有安装 `pytest`，因此没有运行完整 pytest。

## 当前服务器运行状态

服务器运行命令：

```bash
conda run -n dyn python src/examples/run_on_data_feh_global.py \
  --stage core \
  --heldout-fraction 0.2 \
  --heldout-seed 20260812
```

讨论时 Python 进程 PID 为 `3971482`。检查结果：

- 进程持续运行，约占用一个 CPU 核，状态为 `Rl+`。
- JAX 后端确认为 GPU：

```text
backend: gpu
devices: [cuda(id=0), cuda(id=1)]
```

- `/proc/3971482/fd` 同时打开了 `/dev/nvidia0` 和 `/dev/nvidia1`，说明进程已经初始化并连接两张 GPU。
- 因此它不是 CPU fallback，也不像死锁。
- `nvidia-smi` 报 NVML 驱动/库版本不一致：

```text
Failed to initialize NVML: Driver/library version mismatch
NVML library version: 580.126
```

这是 NVML 管理接口问题；当前 JAX CUDA 计算仍然能够运行。

没有实时输出的主要原因是 `conda run` 默认捕获 stdout。下次应使用：

```bash
conda run --no-capture-output -n dyn python -u \
  src/examples/run_on_data_feh_global.py \
  --stage core \
  --heldout-fraction 0.2 \
  --heldout-seed 20260812 2>&1 | tee core.log
```

如果已经执行 `conda activate dyn`，可以直接使用 `python -u`，不必再嵌套 `conda run`。

## 性能方面的重要问题

当前单次 likelihood 的核心计算规模大约为：

```text
12,029 个训练系统
× 11 个 FeH quadrature nodes
× 约 800 个速度积分点
≈ 1.06 亿个网格元素
```

除此之外还有：

- 1200 warmup
- 2000 samples
- 4 chains
- dense mass matrix
- 只有 2 张 GPU

因此即使 GPU 正常，运行也可能很慢。4 条链在 2 张 GPU 上未必能全部同时并行，相关 NumPyro 警告可能被原命令的 `conda run` 捕获。

只要进程 CPU time 持续增加，就应继续等待，不要仅仅为了查看输出而重启。可用以下命令监控：

```bash
watch -n 30 \
  'ps -p 3971482 -o pid,etime,time,%cpu,%mem,rss,stat'
```

PID 只对当时那次运行有效，重新启动后需要用 `pgrep -af run_on_data_feh_global.py` 获取新 PID。

## 下一步

1. 等待 `core` 阶段完成。
2. 用诊断 notebook 检查 divergence、R-hat、bulk/tail ESS、BFMI、acceptance probability 和 tree-depth saturation。
3. 使用完全相同的 held-out seed 运行 `quadratic`。
4. 再运行 `cross`。
5. 比较三个阶段的 held-out `Δlppd ± SE`。
6. 如果 `cross` 没有显著提高 held-out lppd，或者采样诊断明显变差，就不要保留 `c_xy`。
7. 如果运行时间过长，为脚本增加以下 CLI 参数：
   - `--num-chains`
   - `--num-warmup`
   - `--num-samples`
   - `--feh-quadrature-nodes`
   - 速度积分分辨率参数
   - 可选训练子样本数
8. 正式长时间运行前，可以先用 2 chains 和较少 warmup/samples 做性能与模型 smoke run。

