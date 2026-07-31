# Bayesian Binary Masses

本仓库当前的主要入口是
[`src/examples/run_on_data_feh_global.py`](src/examples/run_on_data_feh_global.py)。
它使用真实的 Gaia 宽双星数据，通过 JAX/NumPyro 的 NUTS 采样拟合连续金属丰度
`[Fe/H]` 下的质量–绝对星等关系。

本文档以成功运行该脚本为目标。仓库中的其他示例和旧模型不是当前主要工作流。

## 模型概览

主脚本使用 `DifferencePolyFehMLR`：

- 以 `data/interpolated_mass_data/` 中的等时线质量网格
  `m_iso(M_G, [M/H])` 为基线；
- 使用依赖绝对 G 星等和金属丰度的二维多项式修正 `log10(mass)`；
- 使用 Rice 分布描述观测不确定性；
- 可同时拟合离群点比例及离群点分布参数；
- 使用 NumPyro NUTS 进行后验采样；
- JAX 能使用 GPU 时自动使用 GPU，否则自动使用 CPU。

代码中将观测到的 `[Fe/H]` 直接作为等时线网格的 `[M/H]` 使用，即假设
`[M/H] ≈ [Fe/H]`。

## 仓库中需要的文件

默认配置直接使用仓库内的以下文件：

```text
bayesian-binary-masses/
├── data/
│   ├── jd_msms_single_bic_1kpc_filtered_cutb_fehloss.fits
│   └── interpolated_mass_data/
│       ├── gmag_grid.npy
│       └── mass_interp_MH_*.npy
├── src/
│   ├── binary_masses/
│   │   ├── differencepoly_feh.py
│   │   ├── isochrone_grid.py
│   │   ├── jax_utils.py
│   │   └── polynomial.py
│   └── examples/
│       └── run_on_data_feh_global.py
└── requirements.txt
```

主脚本会根据自身位置计算仓库根目录，因此不依赖启动命令所在的当前目录；不过仍建议
从 `bayesian-binary-masses/` 根目录运行。

## 环境安装

建议使用独立的 Python 3.10 或 3.11 环境：

```bash
cd bayesian-binary-masses

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如需把 `binary_masses` 安装为可编辑包，可继续执行：

```bash
python -m pip install -e .
```

主脚本自身会把 `src/` 加入 Python 路径，所以仅运行该脚本时，完成依赖安装即可。

### GPU 与 CPU

代码不强制指定平台，而是使用 JAX 的自动设备选择：

- JAX 能识别兼容 GPU：使用 GPU；
- 没有可用 GPU：使用 CPU。

运行时会看到类似输出：

```text
NumPyro: running on GPU backend (...)
```

或：

```text
NumPyro: running on CPU backend (1 device(s)).
```

可在当前 Python 环境中提前检查：

```bash
python -c "import jax; print(jax.devices())"
```

需要注意，机器装有 NVIDIA GPU 不等于 JAX 一定能使用它；还需要匹配的驱动和
GPU 版 JAX。GPU 安装方式取决于 CUDA/操作系统版本，请参考
[JAX 官方安装说明](https://docs.jax.dev/en/latest/installation.html)。如果安装的是
CPU 版 JAX，脚本仍能运行，但会使用 CPU，耗时通常明显更长。

## 运行主脚本

在仓库根目录执行：

```bash
python src/examples/run_on_data_feh_global.py
```

默认工作流会：

1. 读取 `data/jd_msms_single_bic_1kpc_filtered_cutb_fehloss.fits`；
2. 保留 `a_g_edhf_1 <= 0.1` 的系统；
3. 使用固定随机种子，从数据中最多抽取 5000 个系统；
4. 根据自行、视差和投影间距计算 `v`、`u` 与 `u_sigma`；
5. 加载连续 `[M/H]` 等时线质量曲面；
6. 运行 NumPyro MCMC；
7. 保存后验样本、corner plot 和多金属丰度质量–星等关系图。

默认 MCMC 设置为 800 个 warmup、6000 个后验样本、1 条 chain，并使用 dense
mass matrix。这是正式拟合配置，首次编译和完整采样可能耗时较长。

### 先做快速测试

如需先确认完整流程可以跑通，可在
[`run_on_data_feh_global.py`](src/examples/run_on_data_feh_global.py) 底部的
`test_differencepoly_feh_model(...)` 调用中临时减小：

```python
num_warmup=100,
num_samples=200,
use_dense_mass=False,
```

也可以把脚本中最多抽取 5000 个系统的限制临时改小。确认数据加载、JAX 编译、采样和
绘图均正常后，再恢复正式设置。

## 输入数据要求

默认 FITS 文件至少需要以下列：

| 列名 | 用途 |
| --- | --- |
| `a_g_edhf_1` | 消光筛选 |
| `pmra1`, `pmra2` | 两成员的赤经方向自行 |
| `pmdec1`, `pmdec2` | 两成员的赤纬方向自行 |
| `parallax1` | 视差 |
| `sep_AU` | 投影间距 |
| `dpm_over_error` | 计算 `u_sigma` |
| `absg1`, `absg2` | 两成员的绝对 G 星等 |
| `feh` | 首选金属丰度列 |

如果不存在 `feh`，脚本会尝试使用 `feh_jcaps_1`。如果两列都不存在，模型设置数据时会
报错。

脚本中的计算为：

```text
v       = 4.74 × sqrt((pmra2-pmra1)² + (pmdec2-pmdec1)²) / parallax1
u       = v × sqrt(sep_AU)
u_sigma = u / dpm_over_error
```

若要使用其他数据文件，请修改 `main()` 中的 `data_path`，并确保列名和单位与上述计算
一致。

## 主要配置

主要参数集中在脚本底部的 `test_differencepoly_feh_model(...)` 调用中：

| 参数 | 当前默认值 | 作用 |
| --- | ---: | --- |
| `feh_min`, `feh_max` | `-1`, `0.6` | 拟合的金属丰度范围 |
| `absg_min`, `absg_max` | `3.5`, `13.5` | 绝对 G 星等范围 |
| `mass_min`, `mass_max` | `0.05`, `2.0` | 质量范围，单位为太阳质量 |
| `order` | `1` | 二维修正模型的默认阶数选择 |
| `uncertainty_model` | `"rice"` | 观测不确定性模型 |
| `fit_outlier_params` | `True` | 是否拟合离群点参数 |
| `num_warmup` | `800` | NUTS warmup 数 |
| `num_samples` | `6000` | 后验样本数 |
| `num_chains` | `1` | MCMC chain 数 |
| `use_dense_mass` | `True` | 是否使用 dense mass matrix |
| `seed` | `33` | NumPyro 随机种子 |
| `anchor_enabled` | `True` | 是否启用太阳质量锚点 |

当前太阳锚点为：

```text
M_G = 4.67, [Fe/H] = 0.0, mass = 1.0 M_sun, sigma = 0.01
```

## 输出

默认输出目录是：

```text
results/data_diffpoly2d_anchor_cute/
```

主要输出包括：

- `mcmc_*.txt`：后验参数样本；
- `corner*.png`：后验参数的 corner plot；
- `fit_multifeh_*.png`：不同 `[Fe/H]` 下的质量–星等关系及其不确定度。

文件名会编码不确定性模型、离群点设置、模型阶数、金属丰度范围和其他关键参数。已有
同名文件会被覆盖，正式运行前请按需备份结果。

## 常见问题

### `ModuleNotFoundError`

确认已经激活创建环境，并在该环境中执行：

```bash
python -m pip install -r requirements.txt
```

使用 `python -m pip` 可以避免把依赖安装到另一个 Python 环境。

### 有 GPU 但程序显示 CPU

执行：

```bash
python -c "import jax; print(jax.devices())"
```

如果这里只显示 CPU，说明当前 Python 环境中的 JAX 没有识别到 GPU。检查 GPU 驱动、
CUDA 与 JAX 安装是否匹配。

### GPU/CPU 内存不足或运行过慢

先减少输入系统数，并降低 `num_warmup` 和 `num_samples` 做流程测试。保持
`num_chains=1`；必要时将 `use_dense_mass=False`。

### 找不到等时线数据

确认以下文件存在：

```text
data/interpolated_mass_data/gmag_grid.npy
data/interpolated_mass_data/mass_interp_MH_*.npy
```

主脚本已经向模型传入绝对路径，通常从其他目录启动也不会影响该路径。

## 许可证

本项目使用 MIT License，详见 [`LICENSE`](LICENSE)。
