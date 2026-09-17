"""Extract the empirical MLR from Hwang et al. 2023 (arXiv:2308.08584).

Loads the pre-trained neural network (HR_mass repo, trained_model_20230809.pth),
locates the single-star main-sequence ridge in the Gaia CMD from the released
per-star mass table (20230810_wb_table_with_mass.fits), and outputs:
  1. hwang2023_mass_map.png   -- NN mass over the (BP-RP, M_G) plane
  2. hwang2023_mlr.png        -- MLR: mass vs BP-RP and mass vs M_G along the MS ridge
  3. hwang2023_mlr.csv        -- the ridge MLR table (NN prediction + per-star medians)
"""
import sys

import numpy as np
import torch
from astropy.table import Table
import matplotlib.pyplot as plt

HR_MASS_DIR = "/tmp/hr_mass"
sys.path.insert(0, HR_MASS_DIR)
from binary_training_v1 import mmodel_2  # noqa: E402

OUT_DIR = "/Users/jdli/Project/collab/ytw/bayesian-binary-masses/src/examples/hwang_mlr"

torch.manual_seed(0)

# ---- load the trained model exactly as in Demo 0 -----------------------------
model = mmodel_2(2, 128, 1, 0.2)
state_dict = torch.load(f"{HR_MASS_DIR}/trained_model_20230809.pth", map_location="cpu")
model.load_state_dict(state_dict)

# sanity check at the solar point (Demo 0 reports 0.961 +/- 0.076 Msun)
model.eval()
with torch.no_grad():
    sun_point = torch.exp(model(torch.tensor([[0.82, 4.67]]))).item()
model.train()  # dropout active -> epistemic sampling
with torch.no_grad():
    sun_samples = torch.exp(model(torch.tensor([[0.82, 4.67]] * 1000))).flatten()
print(f"Sun check: point={sun_point:.3f}  median±std={sun_samples.median():.3f}±{sun_samples.std():.3f} Msun")

def predict(bprp, absg, n_draws=1000):
    """NN mass: eval-mode point estimate + dropout-sampling median/std."""
    x = torch.tensor(np.column_stack([bprp, absg]), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        point = torch.exp(model(x)).flatten().numpy()
    model.train()
    draws = []
    with torch.no_grad():
        for _ in range(n_draws):
            draws.append(torch.exp(model(x)).flatten().numpy())
    draws = np.array(draws)
    return point, np.median(draws, axis=0), draws.std(axis=0)

# ---- per-star measured masses from the released table ------------------------
t = Table.read(f"{HR_MASS_DIR}/20230810_wb_table_with_mass.fits")
bprp = np.concatenate([t["bp_rp1"], t["bp_rp2"]])
absg = np.concatenate([t["absg1"], t["absg2"]])
mass = np.concatenate([t["predict_mass1_median"], t["predict_mass2_median"]])
print(f"stars: {len(bprp)}, bp_rp range {bprp.min():.2f}-{bprp.max():.2f}, "
      f"absg range {absg.min():.2f}-{absg.max():.2f}")

# ---- empirical single-star MS ridge: peak of the absg distribution per color --
edges_c = np.arange(0.55, 3.95, 0.10)
centers_c = 0.5 * (edges_c[:-1] + edges_c[1:])
edges_g = np.arange(2.0, 15.01, 0.10)
ridge_g, ridge_n = [], []
for lo, hi in zip(edges_c[:-1], edges_c[1:]):
    s = (bprp >= lo) & (bprp < hi) & (absg > 2) & (absg < 15)
    n = s.sum()
    if n < 200:
        ridge_g.append(np.nan)
        ridge_n.append(0)
        continue
    hist, _ = np.histogram(absg[s], bins=edges_g)
    ridge_g.append(0.5 * (edges_g[:-1] + edges_g[1:])[np.argmax(hist)])
    ridge_n.append(n)
ridge_g = np.array(ridge_g)
ridge_n = np.array(ridge_n)
# 3-bin running median to smooth bin noise
ridge_g_s = np.copy(ridge_g)
for i in range(len(ridge_g)):
    w = ridge_g[max(0, i - 1):i + 2]
    w = w[np.isfinite(w)]
    ridge_g_s[i] = np.median(w) if len(w) else np.nan

# Bins below the n>=200 threshold carry NaN ridges; the running median can
# still fill them from neighbours, so require real stars before export.
ok = np.isfinite(ridge_g_s) & (ridge_n > 0)

# ---- MLR along the ridge ------------------------------------------------------
nn_point, nn_med, nn_err = predict(centers_c[ok], ridge_g_s[ok])

data_med, data_std = np.zeros(ok.sum()), np.zeros(ok.sum())
for i, (c, g) in enumerate(zip(centers_c[ok], ridge_g_s[ok])):
    s = (np.abs(bprp - c) < 0.05) & (np.abs(absg - g) < 0.25)
    data_med[i], data_std[i] = np.median(mass[s]), mass[s].std()

# ---- mass map over the CMD -----------------------------------------------------
gc = np.arange(0.5, 4.51, 0.02)
gg = np.arange(2.5, 16.01, 0.05)
GC, GG = np.meshgrid(gc, gg)
map_point, map_med, map_err = predict(GC.ravel(), GG.ravel(), n_draws=200)
MAP = map_med.reshape(GC.shape)

# ---- figure 1: mass map --------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 8), dpi=150)
h = ax.hist2d(bprp, absg, bins=[np.arange(0.4, 4.7, 0.05), np.arange(4, 16.2, 0.08)],
              cmin=1, cmap="Greys", norm=plt.matplotlib.colors.LogNorm())
cs = ax.contour(GC, GG, MAP, levels=[0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0],
                colors="k", linewidths=0.8)
ax.clabel(cs, fmt="%.1f", fontsize=7)
ax.plot(centers_c[ok], ridge_g_s[ok], "r-", lw=2, label="MS ridge")
ax.plot([0.82], [4.67], "*", ms=14, color="gold", mec="k", label="Sun")
ax.invert_yaxis()
ax.set_xlabel("BP $-$ RP")
ax.set_ylabel("$M_G$")
ax.set_xlim(0.4, 4.6)
ax.set_ylim(16, 4)
ax.legend(frameon=False, loc="upper right")
fig.colorbar(h[3], ax=ax, label="stars per bin")
ax.set_title("Hwang et al. 2023 NN mass (contours, $M_\\odot$)")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/hwang2023_mass_map.png")
plt.close(fig)

# ---- figure 2: the MLR ----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
ax = axes[0]
ax.errorbar(centers_c[ok], data_med, yerr=data_std, fmt=".", ms=4, alpha=0.4,
            color="0.6", label="per-star medians (ridge window)")
ax.errorbar(centers_c[ok], nn_med, yerr=nn_err, fmt="k-", lw=1.5, capsize=2,
            label="NN on the ridge")
ax.set_xlabel("BP $-$ RP")
ax.set_ylabel("mass ($M_\\odot$)")
ax.legend(frameon=False, loc="lower left", fontsize=9)
ax.set_title("Hwang et al. 2023 dynamical MLR")
ax = axes[1]
o = np.argsort(ridge_g_s[ok])
ax.errorbar(ridge_g_s[ok][o], nn_med[o], yerr=nn_err[o], fmt="k-", lw=1.5, capsize=2)
ax.plot([4.67], [sun_point], "*", ms=14, color="gold", mec="k", label=f"Sun: {sun_point:.2f} $M_\\odot$")
ax.set_xlabel("$M_G$")
ax.set_ylabel("mass ($M_\\odot$)")
ax.invert_xaxis()
ax.legend(frameon=False)
ax.set_title("same MLR vs $M_G$")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/hwang2023_mlr.png")
plt.close(fig)

# ---- CSV ------------------------------------------------------------------------
out = Table()
out["bp_rp"] = centers_c[ok]
out["M_G_ridge"] = ridge_g_s[ok]
out["mass_nn"] = nn_med
out["mass_nn_err"] = nn_err
out["mass_data_median"] = data_med
out["mass_data_std"] = data_std
out["n_stars"] = ridge_n[ok]
out.write(f"{OUT_DIR}/hwang2023_mlr.csv", overwrite=True)

print("MLR (BP-RP, M_G, NN mass±err, data median±std):")
for r in out[::4]:
    print(f"  {r['bp_rp']:.2f}  {r['M_G_ridge']:5.2f}  {r['mass_nn']:.3f}±{r['mass_nn_err']:.3f}"
          f"   {r['mass_data_median']:.3f}±{r['mass_data_std']:.3f}  (n={r['n_stars']})")
print(f"wrote {OUT_DIR}/hwang2023_mlr.csv, hwang2023_mlr.png, hwang2023_mass_map.png")
