# QNHD-Fold: Quantum-Neural Hybrid Diffusion for Protein Folding

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/QNHD-Fold/blob/main/QNHD_Fold_Colab.ipynb)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Quantum-neural hybrid approach for protein structure prediction**

**Author:** Tommaso Marena  
**Institution:** The Catholic University of America  
**Project:** Quantum Protein Folding Research

---

## 🚀 Quick Start

### Try it Now in Google Colab (No Installation Required!)

Click the badge above or [open the notebook](https://colab.research.google.com/github/Tommaso-R-Marena/QNHD-Fold/blob/main/QNHD_Fold_Colab.ipynb) to run QNHD-Fold instantly in your browser.

### Local Installation

```bash
git clone https://github.com/Tommaso-R-Marena/QNHD-Fold.git
cd QNHD-Fold
pip install -r requirements.txt
```

### Basic Usage

```python
from qnhd_fold import QNHDFold

# Initialize model
model = QNHDFold()

# Predict structure
sequence = "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF"
structure = model.predict_structure(sequence)

# Access results
print(f"Mean confidence: {structure.confidence.mean():.1f}")
structure.save("predicted.pdb")
```

---

## 🎯 Overview

QNHD-Fold combines **quantum computing** principles with **deep learning** diffusion models for protein structure prediction.

### Key Components

1. **Quantum Energy Landscape**
   - VQE-inspired energy calculations
   - Hamiltonian eigenvalue spectrum
   - Physical grounding for conformational sampling

2. **Evolution-Guided Diffusion**
   - MSA-like evolutionary features
   - Coevolution matrix encoding
   - Conservation signal integration

3. **Pairformer Encoder**
   - Triangular attention mechanism
   - Geometric constraint enforcement
   - Pair representation learning

4. **Dual-Score Fusion** ⭐ *Novel*
   - Dynamic blending of quantum + neural gradients
   - `λ(t) = 0.3 × (1 - t/T)`
   - Early: quantum exploration, Late: neural refinement

5. **Multi-Modal Confidence**
   - Per-residue pLDDT scores (0-100)
   - Pairwise PAE matrix
   - Epistemic uncertainty quantification

---

## 🔬 Technical Details

### Architecture

```
Input Sequence
    ↓
Evolutionary Features → Pairformer Encoder
    ↓                        ↓
Quantum Landscape    Pair Representations
    ↓                        ↓
    └──→ Dual-Score Fusion ←──┘
              ↓
        Diffusion Model
              ↓
    3D Structure + Confidence
```

### Diffusion Process

**Forward diffusion:** `x₀ (folded) → x_T (noise)`

**Reverse diffusion:** `x_T → x₀ (predicted)`

At each timestep `t`:
```python
neural_score = -∇_x log p(x_t | features)
quantum_score = -∇_x E_quantum(x_t)
fused_score = (1-λ(t))·neural + λ(t)·quantum
x_{t-1} = denoise(x_t, fused_score)
```

### Mathematical Formulation

**Quantum Hamiltonian:**
```
H = H_backbone + H_sidechain + H_contact + H_torsion
```

**Score Fusion:**
```
λ(t) = 0.3 × (1 - t/T)
s_fused = (1-λ)·s_neural + λ·s_quantum
```

---

## 📊 Benchmarks

Performance on various protein types:

| Target Type | Example | Typical pLDDT |
|------------|---------|---------------|
| Small globular | Villin headpiece | 85-92 |
| Alpha-helical | Coiled coil | 88-94 |
| Beta-sheet | Immunoglobulin | 82-89 |
| Mixed α/β | TIM barrel | 85-91 |

**Runtime:** ~2 minutes for 50 residues (Google Colab GPU)

---

## 📁 Repository Structure

```
QNHD-Fold/
├── README.md                          # This file
├── TECHNICAL_GUIDE.md                 # Detailed technical documentation
├── requirements.txt                   # Python dependencies
├── QNHD_Fold_Colab.ipynb             # Interactive Google Colab notebook
├── qnhd_fold/                        # Core implementation
│   ├── __init__.py
│   ├── model.py                      # Main QNHD-Fold model
│   ├── quantum_circuits.py           # Quantum energy module
│   ├── diffusion.py                  # Dual-score diffusion
│   ├── encoder.py                    # Pairformer encoder
│   ├── confidence.py                 # Confidence prediction
│   └── experimental_apis.py          # PDB/UniProt integration
├── benchmarks/                       # Evaluation scripts
│   ├── casp_benchmark.py
│   └── membrane_protein_benchmark.py
├── data/                            # Benchmark results
│   ├── QNHD_real_pdb_benchmark.csv
│   ├── QNHD_membrane_protein_results.csv
│   └── QNHD_idr_results.csv
├── examples/                        # Usage examples
│   └── basic_prediction.py
└── docs/                           # Documentation
    └── USAGE_GUIDE.md
```

---

## 🛠️ Advanced Usage

### Custom Configuration

```python
from qnhd_fold import QNHDFold, DiffusionConfig

config = DiffusionConfig(
    num_timesteps=1000,
    noise_schedule="cosine",
    beta_start=1e-4,
    beta_end=0.02
)

model = QNHDFold(config=config)

structure = model.predict_structure(
    sequence,
    num_diffusion_steps=200,  # More steps = higher accuracy
    verbose=True
)
```

### Quantum Circuit Integration

```python
from qnhd_fold.quantum_circuits import QuantumProteinCircuit

quantum = QuantumProteinCircuit(n_qubits=12)
energy = quantum.compute_energy(coordinates)
```

### API Integration

```python
from qnhd_fold.experimental_apis import ExperimentalDataAPI

api = ExperimentalDataAPI()

# Fetch from PDB
pdb_data = api.fetch_pdb_structure('1UBQ')

# Fetch from UniProt
uniprot_data = api.fetch_uniprot_entry('P0DTC2')
```

---

## 📖 Documentation

- **[Technical Guide](TECHNICAL_GUIDE.md)** - Complete technical documentation
- **[Usage Guide](docs/USAGE_GUIDE.md)** - Detailed usage examples
- **[Colab Notebook](QNHD_Fold_Colab.ipynb)** - Interactive tutorial

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_diffusion.py

# With coverage
pytest --cov=qnhd_fold tests/
```

---

## 📦 Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
scipy>=1.7.0
```

**Optional:**
```
pennylane>=0.32.0  # For quantum circuits
qiskit>=0.45.0     # Alternative quantum backend
biopython>=1.79    # For PDB I/O
```

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**Tommaso Marena**  
The Catholic University of America  
Email: tmarena@cua.edu  
GitHub: [@Tommaso-R-Marena](https://github.com/Tommaso-R-Marena)

---

## 🙏 Acknowledgments

- The Catholic University of America
- Dr. Dominick Rizk
- Dr. Daniel Sierra-Sosa
- Sandeep Shiraskar

---

## 📚 Citation

```bibtex
@software{marena2026qnhd,
  author = {Marena, Tommaso},
  title = {QNHD-Fold: Quantum-Neural Hybrid Diffusion for Protein Folding},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/Tommaso-R-Marena/QNHD-Fold}
}
```

---

**Last Updated:** March 2026

## Backend Runtime

QNHD-Fold now includes a **torch-native backend** for encoder/diffusion/quantum modules when PyTorch is available, with a NumPy fallback for lightweight environments. This enables seamless migration to GPU and mixed-precision workflows without breaking CPU-only usage.
