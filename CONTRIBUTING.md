# Contributing to Bayesian Binary Mass Estimation

We welcome contributions to the Bayesian Binary Mass Estimation package! This document provides guidelines for contributing to the project.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Familiarity with Bayesian inference and astronomical data analysis

### Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/yourusername/bayesian-binary-masses.git
   cd bayesian-binary-masses
   ```

2. **Create a development environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode with dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks (optional but recommended):**
   ```bash
   pre-commit install
   ```

## Running Tests

The project uses pytest for testing:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=binary_masses --cov-report=html

# Run specific test file
pytest tests/test_core.py
```

## Code Style

We use the following tools to maintain code quality:

- **Black** for code formatting
- **flake8** for linting
- **isort** for import sorting

To format your code:

```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
```

## Types of Contributions

We welcome several types of contributions:

### 1. Bug Reports

Please file bug reports using the GitHub issue tracker with:

- Clear description of the problem
- Minimal reproducible example
- Expected vs. actual behavior
- Environment details (Python version, package versions)

### 2. Feature Requests

Feature requests should include:

- Clear description of the proposed feature
- Use case and motivation
- Potential implementation approach (if known)

### 3. Code Contributions

#### Adding New Features

1. Create a new branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Add tests for your new functionality
3. Implement your changes
4. Update documentation if needed
5. Ensure all tests pass
6. Submit a pull request

#### Fixing Bugs

1. Create a branch with a descriptive name:
   ```bash
   git checkout -b fix/bug-description
   ```

2. Add a test that reproduces the bug
3. Fix the issue
4. Ensure all tests pass
5. Submit a pull request

### 4. Documentation

Documentation contributions are valuable! You can help by:

- Improving docstrings
- Adding examples to the README
- Creating tutorials
- Fixing typos or unclear explanations

## Development Guidelines

### Code Organization

- Follow the existing directory structure
- Keep functions and classes focused on single responsibilities
- Use clear, descriptive names
- Add type hints where appropriate

### Testing

- Write tests for all new functionality
- Aim for high test coverage
- Use pytest fixtures for common setup
- Test edge cases and error conditions

### Documentation

- Write clear docstrings following the NumPy style
- Include parameter types, descriptions, and examples
- Document all public APIs
- Keep README and examples up to date

### Performance Considerations

- Profile code changes that might affect performance
- Consider memory usage for large datasets
- Use JAX/NumPy optimizations where appropriate
- Document any performance trade-offs

## Submitting Pull Requests

1. **Create a descriptive title** for your pull request
2. **Reference relevant issues** (e.g., "Fixes #123")
3. **Provide a clear description** of your changes
4. **Include screenshots** if your changes affect visualizations
5. **Ensure CI passes** (automated tests and checks)
6. **Request code review** from maintainers

## Review Process

Maintainers will review your pull request and may request:

- Code changes
- Additional tests
- Documentation updates
- Performance improvements

Please be responsive to feedback and work collaboratively to improve the contribution.

## Release Process

Maintainers handle releases following semantic versioning:

- **MAJOR**: Breaking changes
- **MINOR**: New features (backwards compatible)
- **PATCH**: Bug fixes (backwards compatible)

## Community Guidelines

- Be respectful and constructive
- Welcome newcomers and help them learn
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Help

If you need help contributing:

- Check existing issues and discussions
- Start a new discussion on GitHub
- Contact maintainers directly
- Consult the documentation

## Attribution

Contributors will be acknowledged in:
- AUTHORS file
- Release notes
- Documentation

Thank you for contributing to Bayesian Binary Mass Estimation!