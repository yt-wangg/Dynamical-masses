# GitHub Repository Setup Guide

This guide will help you set up your "Dynamical masses" GitHub repository with the professional structure we've created.

## Step 1: Initialize Local Repository

First, navigate to your repository directory and initialize Git:

```bash
cd /home/wyt/Dyn/bayesian-binary-masses
git init
```

## Step 2: Create GitHub Repository

1. Go to [GitHub](https://github.com) and sign in
2. Click the "+" icon in the top right corner and select "New repository"
3. Fill in the repository details:
   - **Repository name**: `Dynamical-masses`
   - **Description**: `Bayesian inference for dynamical masses of binary stars with Rice distribution uncertainties`
   - **Visibility**: Choose Public or Private as needed
   - **Don't** initialize with README (we already have one)
   - **Don't** add .gitignore (we already have one)
   - **Don't** add license (we already have one)

4. Click "Create repository"
5. GitHub will show you commands to push an existing repository

## Step 3: Configure Local Repository

Add the GitHub repository as remote:

```bash
git remote add origin https://github.com/yourusername/Dynamical-masses.git
```

## Step 4: Prepare Initial Commit

Add all files and create initial commit:

```bash
git add .
git commit -m "Initial commit: Bayesian binary mass estimation package

- Add core Bayesian inference with Rice distribution uncertainties
- Implement non-parametric mass-luminosity relation fitting
- Support multi-metallicity analysis with NumPyro
- Include comprehensive examples and documentation
- Add test suite and development tools

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>"
```

## Step 5: Push to GitHub

Push your code to GitHub:

```bash
git branch -M main
git push -u origin main
```

## Step 6: Configure Repository Settings

### Repository Description and Topics

On GitHub, go to your repository → Settings → General:

- **About section**: Already filled, but you can enhance it
- **Topics**: Add these tags:
  ```
  astronomy, astrophysics, bayesian-inference, binary-stars, mcmc, stellar-masses, rice-distribution, numpyro, jax
  ```

### Enable Features

Go to Settings → Options → Features:

- ✅ **Issues** (for bug reports and feature requests)
- ✅ **Projects** (if you want project boards)
- ✅ **Wikis** (optional, for additional documentation)
- ✅ **Security and analysis** (for dependency alerts)
- ✅ **Discussions** (for community questions)

### Branch Protection

Go to Settings → Branches → Add branch protection rule:

- **Branch name pattern**: `main`
- ✅ **Require pull request reviews before merging**
- ✅ **Require approvals**: 1
- ✅ **Dismiss stale PR approvals when new commits are pushed**
- ✅ **Require review from CODEOWNERS** (if you add one)
- ✅ **Require status checks to pass before merging**
- ✅ **Require branches to be up to date before merging**

## Step 7: Set Up GitHub Actions (CI/CD)

Create `.github/workflows/test.yml`:

```yaml
name: Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: [3.8, 3.9, '3.10', 3.11]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -e .

    - name: Run tests
      run: |
        pytest tests/ -v --cov=binary_masses

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
```

## Step 8: Create Additional Documentation Files

### AUTHORS file

Create `AUTHORS`:

```
# Authors of Bayesian Binary Mass Estimation

Yutong Wang <yutong.wang@yourinstitution.edu> - Creator and maintainer

Contributors:
- (Add names and emails as contributors join)
```

### CHANGELOG file

Create `CHANGELOG.md`:

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-XX-XX

### Added
- Initial release of Bayesian Binary Mass Estimation package
- Core Bayesian inference with Rice distribution uncertainties
- Non-parametric mass-luminosity relation fitting
- Multi-metallicity analysis support
- Comprehensive examples and documentation
- Full test suite
- GPU acceleration through JAX/NumPyro
```

## Step 9: Create Documentation Website (Optional)

For comprehensive documentation, consider setting up Read the Docs:

1. Create account on [Read the Docs](https://readthedocs.org)
2. Connect your GitHub repository
3. Add `docs/` directory with Sphinx configuration
4. Add `requirements-docs.txt` for documentation dependencies

## Step 10: Create Release

Once everything is set up:

1. Go to GitHub repository → Releases
2. Click "Create a new release"
3. **Tag version**: `v1.0.0`
4. **Release title**: `Version 1.0.0`
5. **Release notes**: Copy from CHANGELOG.md
6. Click "Publish release"

## Step 11: Install and Test

Test your package installation:

```bash
# Install from GitHub
pip install git+https://github.com/yourusername/Dynamical-masses.git

# Or clone and install in development mode
git clone https://github.com/yourusername/Dynamical-masses.git
cd Dynamical-masses
pip install -e .
```

Test the installation:

```python
from binary_masses import MultiMetallicityFitter, PU8Sampler
print("Package installed successfully!")

# Test basic functionality
sampler = PU8Sampler()
samples = sampler.sample(n_samples=10)
print(f"Generated {len(samples)} samples")
```

## Step 12: Collaborate with Others

To work with collaborators:

1. **Add collaborators**:
   - Settings → Manage access → Invite a collaborator
   - Add your collaborators' GitHub usernames

2. **Set up code review process**:
   - Require pull requests for all changes
   - Use the CODEOWNERS file for required reviewers
   - Set up GitHub Actions to run tests on PRs

3. **Communication**:
   - Use GitHub Issues for bug reports and feature requests
   - Use GitHub Discussions for questions and community discussions
   - Use pull requests for code contributions

## Best Practices

1. **Regular commits**: Commit frequently with clear messages
2. **Branch protection**: Protect your main branch
3. **Automated testing**: Keep CI/CD running
4. **Documentation**: Keep README and docs updated
5. **Version control**: Use semantic versioning
6. **Security**: Enable GitHub's security features
7. **Community**: Be responsive to issues and PRs

## Next Steps

1. Write your first research paper using this package
2. Share with collaborators and get feedback
3. Present at conferences or seminars
4. Consider publishing as a package (e.g., on JOSS)
5. Maintain and update based on user feedback

Your professional repository is now ready for collaborative research!