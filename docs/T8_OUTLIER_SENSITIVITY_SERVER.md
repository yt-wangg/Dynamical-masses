# T8 outlier 形状敏感性：服务器运行说明

这个实验固定 T8a 金属丰度后验，只改变动力学 lookup 中 outlier 分布的中心和宽度，然后分别重跑 lookup 与 T8b MLR。三组参数是

| case | `--outlier-u0` | `--outlier-sigma` |
|---|---:|---:|
| `mu30_sigma13` | 30 | 13 |
| `mu40_sigma20` | 40 | 20 |
| `mu30_sigma20` | 30 | 20 |

结果写入新的 `results/t8_outlier_sensitivity_20260916/<case>/`，不会覆盖 `results/hierarchical_metallicity_t8_20260913/`。outlier 仍采用固定 `[0,80]` 支持区间；本实验只改变中心和宽度，因此 `(30,20)` 不等同于完整复现 Hwang et al. 的模型。lookup 使用 1024 个几何质量尺度节点、`K=14` 的速度窗口和 `R=128` 个局部积分节点；MCMC 每组为 4 chains、每条链 1000 次 warmup 和 1000 个保留样本，seed 为 `20260819`，`target_accept=0.9`。

## 需要上传的文件

把下面五个文件上传到服务器项目的对应路径：

```text
src/binary_masses/hierarchical_metallicity.py
src/examples/run_hierarchical_metallicity_test.py
scripts/run_t8_outlier_sensitivity.sh
docs/T8_OUTLIER_SENSITIVITY_SERVER.md
test/test_t8_outlier_sensitivity.py
```

前两个文件修改模型与运行入口，shell 脚本负责启动三组实验，最后两个文件提供说明和小型数值验证。服务器上已有的 T8 baseline 结果、输入 FITS、`dyn` 环境和项目其他依赖不需要重复上传。若服务器项目根目录是 `/home/wyt/Dyn/Dynamical-masses`，以下命令均从该目录执行。

## 直接运行

```bash
cd /home/wyt/Dyn/Dynamical-masses
conda activate dyn
chmod +x scripts/run_t8_outlier_sensitivity.sh
conda run -n dyn python test/test_t8_outlier_sensitivity.py

export DATA=/home/wyt/Dyn/Dynamical-masses/data/jd_msms_single_bic_1kpc_filtered_cmdcut_cutb_jcaps_err_mhcal_good.fits
export BASELINE_DIR=/home/wyt/Dyn/Dynamical-masses/results/hierarchical_metallicity_t8_20260913
export OUTPUT_ROOT=/home/wyt/Dyn/Dynamical-masses/results/t8_outlier_sensitivity_20260916
export REQUIRE_GPU=1
bash scripts/run_t8_outlier_sensitivity.sh
```

默认 `STAGE=both`、`CASE=all`，会依次完成三组的 lookup 和 MLR。脚本直接引用 `BASELINE_DIR/latent_metallicity_weights_t8.npz`，不会复制或重新拟合金属丰度后验。

## SLURM 提交

脚本自身只放通用资源声明；partition、account 和 GPU 资源由集群命令提供。建议每组提交一个作业，以免三组顺序拟合超过脚本的两天时限：

```bash
cd /home/wyt/Dyn/Dynamical-masses
export DATA=/home/wyt/Dyn/Dynamical-masses/data/jd_msms_single_bic_1kpc_filtered_cmdcut_cutb_jcaps_err_mhcal_good.fits
export BASELINE_DIR=/home/wyt/Dyn/Dynamical-masses/results/hierarchical_metallicity_t8_20260913
export OUTPUT_ROOT=/home/wyt/Dyn/Dynamical-masses/results/t8_outlier_sensitivity_20260916

for case_name in mu30_sigma13 mu40_sigma20 mu30_sigma20; do
  sbatch --partition=PARTITION --account=ACCOUNT --gres=gpu:1 \
    --export=ALL,STAGE=both,CASE="$case_name",CONDA_ENV=dyn,REQUIRE_GPU=1 \
    scripts/run_t8_outlier_sensitivity.sh
done
```

如果集群不使用 GPU，去掉 `--gres=gpu:1` 并设置 `REQUIRE_GPU=0`。不要把旧的 `run_t8_hierarchical_metallicity.slurm` 作为 `SCRIPT` 传给本脚本；这里的提交入口就是新的 shell 脚本。

## 单组或中断后续跑

例如只跑 `mu30_sigma20`：

```bash
CASE=mu30_sigma20 STAGE=both bash scripts/run_t8_outlier_sensitivity.sh
```

如果 lookup 已经完成而 MLR 中断：

```bash
CASE=mu30_sigma20 STAGE=mlr bash scripts/run_t8_outlier_sensitivity.sh
```

如果只需重建 lookup：

```bash
CASE=mu30_sigma20 STAGE=lookup bash scripts/run_t8_outlier_sensitivity.sh
```

脚本发现已有 lookup 或 MLR 结果及其诊断、单调性和派生网格文件时，会核对已保存的 outlier 参数与积分设置，吻合才跳过该阶段；要重新计算某一组，先将该组对应的输出目录移到安全的备份位置，再显式提交相应 stage。MLR 没有 sampler checkpoint，`STAGE=mlr` 会从头完成该组 MLR 拟合。每个 case 的关键文件是 `dynamics_likelihood_lookup_t8.npz`、`lookup_convergence_t8.json`、`mlr_mcmc_t8.npz`、`mlr_diagnostics_t8.json`、`mlr_monotonicity.json`、`mlr_summary_t8.csv` 和 `mlr_derived_grid_t8.npz`。
