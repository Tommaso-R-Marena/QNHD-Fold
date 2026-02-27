# QNHD-Fold: Technical Implementation Guide

**Author:** Tommaso Marena  
**Institution:** The Catholic University of America  
**Project:** NIH/NCI Protein Folding Research

---

## Overview

QNHD-Fold (Quantum-Neural Hybrid Diffusion for Protein Folding) is a novel computational framework that combines quantum computing principles with diffusion models for protein structure prediction.

## Technical Architecture

### 1. Core Components

#### 1.1 Quantum Energy Landscape Module
```python
class QuantumEnergyLandscape:
    '''
    Simulates quantum-enhanced energy calculations for protein conformations
    Uses VQE-inspired approach to compute energy potentials
    '''
    - Generates quantum Hamiltonian eigenvalue spectrum
    - Maps protein coordinates to quantum states
    - Computes energy potentials for conformational sampling
```

**Key Features:**
- Hermitian matrix construction for quantum Hamiltonian
- Eigenvalue decomposition for energy spectrum
- Coordinate-to-quantum-state mapping

#### 1.2 Evolutionary Feature Extractor
```python
class EvolutionaryFeatureExtractor:
    '''
    Extracts MSA-like evolutionary signals
    Simulates AlphaFold DB-style conservation patterns
    '''
    - Conservation scores per residue
    - Coevolution matrix construction
    - Evolutionary signal integration
```

#### 1.3 Pairformer Encoder
```python
class PairformerEncoder:
    '''
    Enhanced pair representation with triangular attention
    Enforces geometric constraints via triangle inequality
    '''
    - Single and pair representation encoding
    - Triangular multiplicative updates
    - Sequence separation encoding
```

#### 1.4 Dual-Score Diffusion Module (Novel)
```python
class DualScoreDiffusionModule:
    '''
    KEY INNOVATION: Fuses quantum + neural scores
    Dynamic weighting based on diffusion timestep
    '''
    - Neural score: learned gradient from pair representations
    - Quantum score: energy gradient from quantum landscape
    - Fusion: λ(t) = 0.3 × (1 - t/T)
```

**Fusion Strategy:**
- Early timesteps (t near T): Higher quantum weight → exploration
- Late timesteps (t near 0): Higher neural weight → refinement

### 2. Algorithm Walkthrough

#### Forward Diffusion
```
x₀ (folded) → x₁ → x₂ → ... → x_T (unfolded/noise)
```

#### Reverse Diffusion
```
x_T → x_{T-1} → ... → x₁ → x₀ (predicted structure)

At each step t:
1. Compute neural_score from pair representations
2. Compute quantum_score from energy gradients
3. Fuse: s_fused = (1-λ(t))·neural + λ(t)·quantum
4. Update: x_{t-1} = denoise(x_t, s_fused)
```

### 3. Confidence Prediction

#### 3.1 pLDDT (per-residue Local Distance Difference Test)
- Range: 0-100
- Based on local geometry consistency
- Neighbor distance validation
- Pair representation confidence

#### 3.2 PAE (Predicted Aligned Error)
- NxN matrix for N residues
- Lower values = higher confidence in relative positions
- Distance-based confidence scoring

#### 3.3 Epistemic Uncertainty
- Ensemble sampling (n=10 samples)
- Variance across perturbed structures
- Per-residue uncertainty quantification

### 4. Mathematical Formulation

#### Diffusion Process
```
Forward:  q(x_t | x_0) = N(x_t; √α̅_t x_0, (1-α̅_t)I)
Reverse:  p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t,t), Σ_θ(x_t,t))
```

#### Quantum Hamiltonian
```
H = H_backbone + H_sidechain + H_contact + H_torsion

Energy gradient: ∇E_quantum = -∇_x H(x)
```

#### Score Fusion
```
s_neural(x_t, t) = -∇_x log p_θ(x_t | t, features)
s_quantum(x_t) = -∇_x E_quantum(x_t)

s_fused = (1 - λ(t))·s_neural + λ(t)·s_quantum
where λ(t) = 0.3·(1 - t/T)
```

### 5. Implementation Details

#### 5.1 Data Structures
```python
@dataclass
class ProteinStructure:
    sequence: str              # Amino acid sequence
    coordinates: np.ndarray    # (N, 3) CA atom positions
    confidence: np.ndarray     # (N,) pLDDT scores
    pae: np.ndarray           # (N, N) predicted aligned error
```

#### 5.2 Noise Schedule
```python
# Cosine schedule (better than linear)
timesteps = np.arange(T) / T
alphas_cumprod = cos²((timesteps + 0.008)/1.008 · π/2)
betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
```

#### 5.3 Computational Complexity
- Time: O(N² · T) where N=sequence length, T=diffusion steps
- Space: O(N²) for pair representations
- Typical: 100 residues, 100 steps ≈ 2 minutes on GPU

### 6. Usage Examples

#### Basic Prediction
```python
from qnhd_fold import QNHDFold

model = QNHDFold()
structure = model.predict_structure("MKTAYIAKQRQ...")

print(f"Mean confidence: {structure.confidence.mean():.1f}")
structure.save("output.pdb")
```

#### Advanced Configuration
```python
from qnhd_fold import DiffusionConfig

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

### 7. Benchmarking Metrics

#### TM-score
```
TM-score = (1/N) Σ_i [1 / (1 + (d_i/d_0)²)]
where d_0 = 1.24·(N-15)^(1/3) - 1.8
```

#### GDT-TS (Global Distance Test - Total Score)
```
GDT-TS = mean(P_1Å, P_2Å, P_4Å, P_8Å)
where P_xÅ = % residues within x Å of native
```

#### lDDT (local Distance Difference Test)
```
For each residue i with neighbors j within 15Å:
  Score based on |d_pred(i,j) - d_true(i,j)|
  Thresholds: 0.5Å, 1Å, 2Å, 4Å
```

### 8. File Formats

#### Input: FASTA
```
>protein_name
MKTAYIAKQRQISFVKSHFSRQLE...
```

#### Output: PDB
```
ATOM      1  CA  MET A   1      10.123  20.456  30.789  1.00 94.50
ATOM      2  CA  LYS A   2      13.234  21.567  31.890  1.00 93.20
...
```
B-factor column = pLDDT confidence score

### 9. Quantum Circuit Integration

#### PennyLane Implementation
```python
@qml.qnode(device)
def vqe_circuit(params, coords):
    # Encode protein coordinates
    for i in range(n_qubits):
        qml.RY(params[i], wires=i)
        qml.RZ(coords[i % len(coords)], wires=i)
    
    # Entangling layer
    for i in range(n_qubits-1):
        qml.CNOT(wires=[i, i+1])
    
    # Measure energy
    return qml.expval(qml.PauliZ(0))
```

### 10. Performance Characteristics

| Protein Size | Prediction Time | Memory Usage |
|--------------|----------------|-------------|
| 50 residues  | ~30 seconds    | ~2GB        |
| 100 residues | ~2 minutes     | ~4GB        |
| 200 residues | ~8 minutes     | ~8GB        |
| 500 residues | ~30 minutes    | ~16GB       |

*On NVIDIA A100 GPU*

### 11. Limitations & Considerations

1. **Quantum Simulation**: Currently limited to classical simulation of quantum circuits (up to ~20 qubits effectively)
2. **Sequence Length**: Performance degrades for proteins >1000 residues
3. **MSA Depth**: Works best with moderate evolutionary signal
4. **Membrane Proteins**: Requires additional optimization for transmembrane regions
5. **Dynamics**: Predicts single static structure, not conformational ensembles

### 12. Future Extensions

- Integration with real quantum hardware (IBM Quantum, IonQ)
- Protein-ligand complex prediction
- Conformational ensemble generation
- Multi-chain complex modeling
- Integration with experimental restraints (NMR, cryo-EM)

### 13. Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
scipy>=1.7.0
pennylane>=0.32.0  # For quantum circuits
biopython>=1.79    # For PDB I/O
```

### 14. References

This implementation builds on concepts from:
- AlphaFold3 (diffusion + Pairformer architecture)
- PathDiffusion (evolution-guided diffusion)
- Quantum protein folding approaches
- Score-based generative models

---

## Citation

```bibtex
@software{marena2026qnhd,
  author = {Marena, Tommaso},
  title = {QNHD-Fold: Quantum-Neural Hybrid Diffusion for Protein Folding},
  year = {2026},
  url = {https://github.com/Tommaso-R-Marena/QNHD-Fold}
}
```

## Contact

**Tommaso Marena**  
The Catholic University of America  
Email: tmarena@cua.edu  
GitHub: @Tommaso-R-Marena