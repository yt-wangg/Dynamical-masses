#!/usr/bin/env python3
"""
Diagnostic 4: Compare BPL vs Non-Parametric Models

This script compares the broken power-law model with the non-parametric model
to determine if the bias is BPL-specific or affects both models.

Expected outcomes:
- If non-parametric is unbiased → problem is BPL-specific (segment logic)
- If both biased similarly → problem is in shared likelihood code (Jacobian)
- If different bias patterns → multiple issues
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from astropy.table import Table

# Add package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from binary_masses import BrokenPowerLawMLR, NonParametricMLR


def load_mock_data(data_path, n_subsample=2000):
    """Load and prepare mock data."""
    print(f"Loading mock data from: {data_path}")
    data = Table.read(data_path)

    # Filter
    data = data[data['u'] > 0]
    data = data[data['absg1'] >= 3]
    data = data[data['absg2'] >= 3]
    data = data[data['feh'] >= -1]
    data = data[data['feh'] <= 0.6]

    # Subsample
    if len(data) > n_subsample:
        np.random.seed(42)
        indices = np.random.choice(len(data), size=n_subsample, replace=False)
        data = data[indices]

    print(f"Using {len(data)} systems")
    return data


def fit_bpl_model(data):
    """Fit BPL model."""
    print(f"\n{'='*70}")
    print("Fitting BPL Model (2 segments)")
    print(f"{'='*70}")

    bpl_fitter = BrokenPowerLawMLR(
        n_segments=2,
        mass_min=0.08,
        mass_max=1.5,
        absg_min=3.0,
        absg_max=14.0,
        uncertainty_model='gaussian',
        f_outlier=0
    )

    bpl_fitter.set_data(
        u_values=data['u'],
        u_sigma_values=data['u_sigma'],
        absg1_values=data['absg1'],
        absg2_values=data['absg2']
    )

    print("Running MCMC...")
    bpl_fitter.run_numpyro(
        num_warmup=500,
        num_samples=1500,
        num_chains=2,
        seed=42,
        a_prior_range=(-20, 10),
        b_prior_range=(-100, 5)
    )

    return bpl_fitter


def fit_nonparametric_model(data):
    """Fit non-parametric model."""
    print(f"\n{'='*70}")
    print("Fitting Non-Parametric Model (10 bins)")
    print(f"{'='*70}")

    nonparam_fitter = NonParametricMLR(
        n_bins=10,
        absg_min=3.0,
        absg_max=14.0,
        mass_min=0.08,
        mass_max=1.5,
        uncertainty_model='gaussian',
        gamma=np.inf,  # No smoothing
        f_outlier=0
    )

    nonparam_fitter.set_data(
        u_values=data['u'],
        u_sigma_values=data['u_sigma'],
        absg1_values=data['absg1'],
        absg2_values=data['absg2']
    )

    print("Running MCMC...")
    nonparam_fitter.run_numpyro(
        num_warmup=500,
        num_samples=1500,
        num_chains=2,
        seed=42
    )

    return nonparam_fitter


def compute_residuals(fitter, data, model_type='bpl'):
    """Compute residuals for a fitted model."""
    if model_type == 'bpl':
        median_params = np.median(fitter.samples, axis=0)
        m1_pred = fitter.mass_from_absg(data['absg1'], median_params)
        m2_pred = fitter.mass_from_absg(data['absg2'], median_params)
    elif model_type == 'nonparam':
        median_masses = np.median(fitter.samples, axis=0)
        m1_pred = np.interp(data['absg1'], fitter.absg_bins, median_masses)
        m2_pred = np.interp(data['absg2'], fitter.absg_bins, median_masses)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    m1_true = np.array(data['m1'])
    m2_true = np.array(data['m2'])

    residuals = np.concatenate([
        (m1_pred - m1_true) / m1_true,
        (m2_pred - m2_true) / m2_true
    ])

    return {
        'residuals': residuals,
        'm1_pred': m1_pred,
        'm2_pred': m2_pred,
        'm1_true': m1_true,
        'm2_true': m2_true,
        'median_bias': np.median(residuals),
        'mean_bias': np.mean(residuals),
        'std_bias': np.std(residuals)
    }


def compare_models(bpl_res, nonparam_res, data, bpl_fitter, nonparam_fitter, output_dir):
    """Create comparison plots and analysis."""
    print(f"\n{'='*70}")
    print("MODEL COMPARISON")
    print(f"{'='*70}")

    # Print statistics
    print(f"\n{'Model':<20} {'Median Bias':<15} {'Mean Bias':<15} {'Std Dev':<15}")
    print("-" * 70)
    print(f"{'BPL':<20} {bpl_res['median_bias']:+.4f} ({bpl_res['median_bias']*100:+.2f}%)  "
          f"{bpl_res['mean_bias']:+.4f}  {bpl_res['std_bias']:.4f}")
    print(f"{'Non-Parametric':<20} {nonparam_res['median_bias']:+.4f} ({nonparam_res['median_bias']*100:+.2f}%)  "
          f"{nonparam_res['mean_bias']:+.4f}  {nonparam_res['std_bias']:.4f}")

    # Differential analysis
    bias_diff = abs(bpl_res['median_bias']) - abs(nonparam_res['median_bias'])
    print(f"\nDifferential Bias (BPL - NonParam): {bias_diff:+.4f}")

    # Create plots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Plot 1: Residual histograms
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(bpl_res['residuals'], bins=40, alpha=0.6, color='blue', label='BPL', edgecolor='black')
    ax1.hist(nonparam_res['residuals'], bins=40, alpha=0.6, color='green', label='Non-Parametric', edgecolor='black')
    ax1.axvline(0, color='red', linestyle='--', linewidth=2)
    ax1.axvline(bpl_res['median_bias'], color='blue', linestyle='--', linewidth=2,
               label=f"BPL median = {bpl_res['median_bias']:+.3f}")
    ax1.axvline(nonparam_res['median_bias'], color='green', linestyle='--', linewidth=2,
               label=f"NonParam median = {nonparam_res['median_bias']:+.3f}")
    ax1.set_xlabel('Fractional Residual', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('Residual Distributions', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Bias comparison bar chart
    ax2 = fig.add_subplot(gs[0, 1])
    models = ['BPL', 'Non-Parametric']
    median_biases = [bpl_res['median_bias'], nonparam_res['median_bias']]
    colors = ['blue', 'green']
    ax2.bar(models, median_biases, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(0, color='black', linestyle='-', linewidth=2)
    ax2.axhline(0.02, color='orange', linestyle='--', linewidth=1, label='±2% threshold')
    ax2.axhline(-0.02, color='orange', linestyle='--', linewidth=1)
    ax2.set_ylabel('Median Fractional Bias', fontsize=11)
    ax2.set_title('Bias Comparison', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Residual boxplot
    ax3 = fig.add_subplot(gs[0, 2])
    bp = ax3.boxplot([bpl_res['residuals'], nonparam_res['residuals']], labels=models, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen']):
        patch.set_facecolor(color)
    ax3.axhline(0, color='red', linestyle='--', linewidth=2)
    ax3.set_ylabel('Fractional Residual', fontsize=11)
    ax3.set_title('Residual Distributions', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: MLR comparison
    ax4 = fig.add_subplot(gs[1, :])
    absg_grid = np.linspace(3, 14, 300)

    # True masses
    ax4.scatter(data['absg1'], data['m1'], s=8, alpha=0.2, color='black', label='True masses', zorder=0)

    # BPL fit
    median_params_bpl = np.median(bpl_fitter.samples, axis=0)
    mass_grid_bpl = bpl_fitter.mass_from_absg(absg_grid, median_params_bpl)
    ax4.plot(absg_grid, mass_grid_bpl, 'b-', linewidth=3, label='BPL fit', zorder=10)

    # Non-parametric fit
    median_masses_nonparam = np.median(nonparam_fitter.samples, axis=0)
    mass_grid_nonparam = np.interp(absg_grid, nonparam_fitter.absg_bins, median_masses_nonparam)
    ax4.plot(absg_grid, mass_grid_nonparam, 'g-', linewidth=3, label='Non-Parametric fit', zorder=10)

    # Non-parametric bin centers
    ax4.scatter(nonparam_fitter.absg_bins, median_masses_nonparam, s=80, color='green',
               marker='o', edgecolor='black', linewidth=1.5, zorder=15, label='NonParam bins')

    # BPL break points
    if hasattr(bpl_fitter, 'break_points') and bpl_fitter.break_points is not None:
        intercepts, slopes = bpl_fitter._split_params(median_params_bpl)
        for i, bp in enumerate(bpl_fitter.break_points):
            bp_mag = intercepts[i] + slopes[i] * np.log10(bp)
            ax4.axvline(bp_mag, color='blue', linestyle='--', linewidth=2, alpha=0.7,
                       label=f'BPL break {i+1}' if i == 0 else '')

    ax4.set_xlabel('Absolute G Magnitude', fontsize=12)
    ax4.set_ylabel('Mass (M$_\\odot$)', fontsize=12)
    ax4.set_title('Mass-Luminosity Relation Comparison', fontsize=14, fontweight='bold')
    ax4.invert_xaxis()
    ax4.legend(fontsize=10, loc='best')
    ax4.grid(True, alpha=0.3)

    # Plot 5: Residuals vs magnitude (BPL)
    ax5 = fig.add_subplot(gs[2, 0])
    absg_all_bpl = np.concatenate([data['absg1'], data['absg2']])
    ax5.scatter(absg_all_bpl, bpl_res['residuals'], alpha=0.3, s=5, color='blue')
    ax5.axhline(0, color='red', linestyle='--', linewidth=2)
    ax5.axhline(bpl_res['median_bias'], color='orange', linestyle='--', linewidth=2)
    ax5.set_xlabel('Absolute G Magnitude', fontsize=11)
    ax5.set_ylabel('Fractional Residual', fontsize=11)
    ax5.set_title('BPL: Residuals vs M_G', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # Plot 6: Residuals vs magnitude (Non-Parametric)
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.scatter(absg_all_bpl, nonparam_res['residuals'], alpha=0.3, s=5, color='green')
    ax6.axhline(0, color='red', linestyle='--', linewidth=2)
    ax6.axhline(nonparam_res['median_bias'], color='orange', linestyle='--', linewidth=2)
    ax6.set_xlabel('Absolute G Magnitude', fontsize=11)
    ax6.set_ylabel('Fractional Residual', fontsize=11)
    ax6.set_title('NonParam: Residuals vs M_G', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # Plot 7: Differential residuals
    ax7 = fig.add_subplot(gs[2, 2])
    diff_residuals = bpl_res['residuals'] - nonparam_res['residuals']
    ax7.hist(diff_residuals, bins=40, alpha=0.7, color='purple', edgecolor='black')
    ax7.axvline(0, color='red', linestyle='--', linewidth=2)
    ax7.axvline(np.median(diff_residuals), color='orange', linestyle='--', linewidth=2,
               label=f"Median = {np.median(diff_residuals):+.3f}")
    ax7.set_xlabel('BPL - NonParam Residual', fontsize=11)
    ax7.set_ylabel('Count', fontsize=11)
    ax7.set_title('Differential Residuals', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)

    output_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {output_path}")
    plt.close()


def main():
    """Main diagnostic routine."""
    print("="*70)
    print("DIAGNOSTIC 4: BPL vs NON-PARAMETRIC MODEL COMPARISON")
    print("="*70)

    data_path = 'data/mock_data_efuncu_unifm_mh_Jsu_n10k_obserr.fits'
    output_dir = 'results/diagnostics'
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found at {data_path}")
        return

    # Load data
    data = load_mock_data(data_path, n_subsample=2000)

    # Fit both models
    bpl_fitter = fit_bpl_model(data)
    nonparam_fitter = fit_nonparametric_model(data)

    # Compute residuals
    bpl_res = compute_residuals(bpl_fitter, data, model_type='bpl')
    nonparam_res = compute_residuals(nonparam_fitter, data, model_type='nonparam')

    # Compare
    compare_models(bpl_res, nonparam_res, data, bpl_fitter, nonparam_fitter, output_dir)

    # Diagnostic conclusion
    print(f"\n{'='*70}")
    print("DIAGNOSTIC CONCLUSION")
    print(f"{'='*70}")

    bpl_bias = abs(bpl_res['median_bias'])
    nonparam_bias = abs(nonparam_res['median_bias'])
    bias_threshold = 0.02

    if bpl_bias > bias_threshold and nonparam_bias < bias_threshold:
        print("⚠️ BPL is biased, but Non-Parametric is NOT biased")
        print("→ Problem is BPL-SPECIFIC")
        print("→ Most likely: Segment selection logic error")
        print("→ Next step: Review diagnostics/segment_selection_test.py results")
    elif bpl_bias > bias_threshold and nonparam_bias > bias_threshold:
        if abs(bpl_bias - nonparam_bias) < 0.01:
            print("⚠️ BOTH models are biased SIMILARLY")
            print("→ Problem is in SHARED likelihood code")
            print("→ Most likely: Jacobian factor issue")
            print("→ Next step: Review diagnostics/jacobian_test.py results")
        else:
            print("⚠️ BOTH models are biased DIFFERENTLY")
            print("→ Multiple issues may be present")
            print("→ BPL has additional bias beyond shared issue")
    elif bpl_bias < bias_threshold and nonparam_bias < bias_threshold:
        print("✓ Both models are unbiased")
        print("→ No systematic bias detected in this test")
        print("→ May need larger sample or different test")
    else:
        print("⚠️ Non-Parametric is biased, but BPL is NOT")
        print("→ Unusual pattern, may indicate non-parametric issue")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
