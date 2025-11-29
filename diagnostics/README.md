# BPL Bias Diagnostic Suite

This directory contains comprehensive diagnostic scripts to identify and fix the systematic mass overestimation bias in the BrokenPowerLawMLR model.

## Problem

The BPL model systematically **overestimates all masses** when fitting mock data, despite recent bug fixes in the integrand and continuity constraints.

## Diagnostic Scripts

### 1. `segment_selection_test.py` (Fast, ~2 min)
**Purpose**: Validate that the segment selection logic correctly inverts the broken power-law relation.

**Tests**:
- Forward-backward consistency: M_G → mass → M_G
- Continuity at break points
- Correct segment assignment

**Expected Outcome**: If all tests pass, segment logic is correct and bias is NOT due to segment selection.

### 2. `residual_analysis.py` (Moderate, ~5-7 min)
**Purpose**: Characterize the bias pattern to determine its nature.

**Analysis**:
- Compute residuals: (predicted - true) / true
- Check for uniform offset vs magnitude-dependent vs mass-scale-dependent
- Generate comprehensive diagnostic plots

**Expected Outcomes**:
- Uniform offset → Jacobian or normalization issue
- Magnitude-dependent → Segment selection issue
- Mass-dependent → Integration grid or scale issue

### 3. `compare_models.py` (Slow, ~10-12 min)
**Purpose**: Compare BPL with Non-Parametric model to determine if bias is BPL-specific or shared.

**Analysis**:
- Fit same data with both models
- Compare residual patterns
- Determine if issue is BPL-specific or in shared likelihood code

**Expected Outcomes**:
- Both biased similarly → Shared likelihood issue (Jacobian)
- Only BPL biased → BPL-specific issue (segment logic)

### 4. `jacobian_test.py` (Slow, ~8-10 min)
**Purpose**: Test whether the `1/sqrt(m_tot)` Jacobian factor is causing the bias.

**Tests**:
- Current implementation (with 1/sqrt_mtot)
- Without Jacobian factor (remove it)
- Inverted Jacobian (sqrt_mtot instead)

**Expected Outcomes**:
- If 'none' has lowest bias → Remove the Jacobian factor
- If 'inverted' has lowest bias → Invert the Jacobian
- If 'current' is best → Jacobian is correct, check elsewhere

## Quick Start

### Option 1: Run All Diagnostics (Recommended)

```bash
cd /Users/jdli/Project/ytw/Dynamical-masses
python diagnostics/run_all_diagnostics.py
```

This will:
1. Run all 4 diagnostic scripts in sequence
2. Generate diagnostic plots in `results/diagnostics/`
3. Provide a comprehensive summary and recommendations

**Estimated time**: 10-15 minutes

### Option 2: Run Individual Diagnostics

```bash
cd /Users/jdli/Project/ytw/Dynamical-masses

# Fast test (2 min)
python diagnostics/segment_selection_test.py

# Characterize bias (5-7 min)
python diagnostics/residual_analysis.py

# Compare models (10-12 min)
python diagnostics/compare_models.py

# Test Jacobian (8-10 min)
python diagnostics/jacobian_test.py
```

## Output

All diagnostic results are saved to:
```
results/diagnostics/
├── residual_analysis_comprehensive.png  # Bias characterization
├── segment_test_forward_backward.png    # Segment logic validation
├── segment_test_continuity.png          # Continuity checks
├── jacobian_comparison.png              # Jacobian factor tests
└── model_comparison.png                 # BPL vs NonParam comparison
```

## Interpreting Results

### If Segment Selection Tests FAIL:
→ **Root cause**: Segment selection logic error
→ **Fix location**: `src/binary_masses/core.py`, lines 1268-1318 (`mass_from_absg_jax_inner`)
→ **Solution**: Simplify segment selection using `jnp.searchsorted`

### If Residual Analysis shows UNIFORM OFFSET:
→ **Root cause**: Likely Jacobian or normalization
→ **Next step**: Review `jacobian_test.py` results
→ **Fix location**: `src/binary_masses/core.py`, lines 1320-1372 (`likelihood_single_u_jax`)

### If Model Comparison shows BOTH models biased:
→ **Root cause**: Shared likelihood code (Jacobian factor)
→ **Next step**: Review `jacobian_test.py` results
→ **Fix location**: `src/binary_masses/core.py`, lines 1358-1366 (integrand calculation)

### If Jacobian Test shows 'none' is best:
→ **Root cause**: Extra Jacobian factor is wrong
→ **Fix**: Remove `(1.0 / sqrt_mtot[:, None])` from integrand
→ **Line**: `src/binary_masses/core.py:1359`

### If Jacobian Test shows 'inverted' is best:
→ **Root cause**: Jacobian is backwards
→ **Fix**: Change `(1.0 / sqrt_mtot[:, None])` to `sqrt_mtot[:, None]`
→ **Line**: `src/binary_masses/core.py:1359`

## Next Steps After Diagnostics

1. **Review diagnostic plots** in `results/diagnostics/`

2. **Identify root cause** from diagnostic outputs

3. **Implement the fix** in `src/binary_masses/core.py`:
   - For Jacobian issue: Modify lines 1320-1372
   - For segment issue: Modify lines 1268-1318

4. **Validate the fix**:
   ```bash
   python diagnostics/residual_analysis.py
   ```
   - Success criterion: Median fractional residual < 0.02

5. **Run comprehensive tests**:
   - Test with different `n_segments` (1, 2, 3)
   - Test with different sample sizes
   - Ensure bias is eliminated universally

## Data Requirements

All scripts expect mock data at:
```
data/mock_data_efuncu_unifm_mh_Jsu_n10k_obserr.fits
```

This FITS file must contain:
- `u`, `u_sigma`: Observed u-parameter and uncertainty
- `absg1`, `absg2`: Absolute G magnitudes
- `m1`, `m2`: **True masses** (for residual calculation)
- `feh`: Metallicity

## Troubleshooting

**Issue**: "Data file not found"
**Solution**: Check that mock data exists at the path above

**Issue**: MCMC convergence warnings
**Solution**: This is OK for diagnostics (using reduced samples for speed)

**Issue**: ImportError for binary_masses
**Solution**: Ensure you run from project root, scripts add src/ to path automatically

**Issue**: JAX/GPU errors
**Solution**: Set `export JAX_PLATFORMS=cpu` before running

## Technical Details

### Computational Cost

- Segment tests: Pure NumPy, very fast
- Other diagnostics: Run MCMC with 500-2000 samples
- Total compute time: ~10-15 minutes on modern CPU

### MCMC Parameters

For speed, diagnostics use:
- 200-500 warmup samples (vs 1000 in production)
- 500-1500 posterior samples (vs 2000 in production)
- 2 chains (vs 4 in production)

This is sufficient to identify systematic bias but may not fully characterize posteriors.

## Implementation Plan

Based on diagnostic findings, the fix will be implemented following this workflow:

1. **Diagnostic Phase** (this directory): Identify root cause
2. **Fix Implementation**: Modify `src/binary_masses/core.py`
3. **Validation**: Re-run diagnostics to confirm bias elimination
4. **Testing**: Run full test suite with production MCMC parameters
5. **Documentation**: Update code comments with explanation

## Contact

For questions about these diagnostics or the BPL bias issue, refer to:
- Main plan: `/Users/jdli/.claude/plans/zany-splashing-penguin.md`
- Test script: `test_bpl_plotting.py`
