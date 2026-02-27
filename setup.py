"""Setup configuration for QNHD-Fold"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="qnhd-fold",
    version="1.0.0",
    author="Tommaso Marena",
    author_email="tmarena@cua.edu",
    description="Quantum-Neural Hybrid Diffusion for Protein Structure Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Tommaso-R-Marena/QNHD-Fold",
    project_urls={
        "Bug Tracker": "https://github.com/Tommaso-R-Marena/QNHD-Fold/issues",
        "Documentation": "https://github.com/Tommaso-R-Marena/QNHD-Fold/wiki",
        "Source Code": "https://github.com/Tommaso-R-Marena/QNHD-Fold",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "quantum": ["pennylane>=0.32.0", "qiskit>=0.45.0"],
        "dev": ["pytest>=7.0.0", "black>=22.0.0", "flake8>=4.0.0"],
        "viz": ["pymol-open-source>=2.5.0", "nglview>=3.0.0"],
    },
)
