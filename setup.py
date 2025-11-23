"""
Setup script for Bayesian Binary Mass Estimation package
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Optional dependencies
extras_require = {
    'dev': [
        'pytest>=6.0.0',
        'black>=21.0.0',
        'flake8>=3.8.0',
        'sphinx>=4.0.0',
        'sphinx-rtd-theme>=0.5.0',
    ],
    'visualization': [
        'seaborn>=0.11.0',
        'plotly>=5.0.0',
    ],
    'all': [
        'pytest>=6.0.0',
        'black>=21.0.0',
        'flake8>=3.8.0',
        'sphinx>=4.0.0',
        'sphinx-rtd-theme>=0.5.0',
        'seaborn>=0.11.0',
        'plotly>=5.0.0',
    ]
}

setup(
    name="bayesian-binary-masses",
    version="1.0.0",
    author="Yutong Wang",
    author_email="yutong.wang@yourinstitution.edu",
    description="Bayesian inference for dynamical masses of binary stars with Rice distribution uncertainties",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/bayesian-binary-masses",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/bayesian-binary-masses/issues",
        "Documentation": "https://bayesian-binary-masses.readthedocs.io/",
        "Source Code": "https://github.com/yourusername/bayesian-binary-masses",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require=extras_require,
    keywords=[
        "astronomy",
        "astrophysics",
        "binary stars",
        "bayesian inference",
        "mcmc",
        "stellar masses",
        "rice distribution",
    ],
    include_package_data=True,
    zip_safe=False,
)