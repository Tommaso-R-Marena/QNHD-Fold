# QNHD-Fold Usage Guide

## Basic Usage

```python
from qnhd_fold import QNHDFold

# 1. Initialize model
model = QNHDFold()

# 2. Predict structure
sequence = "MKTAYIAKQRQISFVKSHFSRQLE"
structure = model.predict_structure(sequence)

# 3. Access results
print(f"Coordinates: {structure.coordinates.shape}")
print(f"Mean confidence: {structure.confidence.mean():.1f}")
print(f"PAE shape: {structure.pae.shape}")

# 4. Save to PDB
structure.save("output.pdb")
```

## Advanced Options

```python
# Customize diffusion parameters
from qnhd_fold import QNHDFold, DiffusionConfig

config = DiffusionConfig(
    num_timesteps=1000,
    noise_schedule="cosine",
    beta_start=1e-4,
    beta_end=0.02
)

model = QNHDFold(config=config)

# More diffusion steps for higher accuracy
structure = model.predict_structure(
    sequence,
    num_diffusion_steps=200,  # default: 100
    verbose=True
)
```

## Quantum Circuit Customization

```python
from qnhd_fold.quantum_circuits import QuantumProteinCircuit

# Initialize quantum module
quantum = QuantumProteinCircuit(n_qubits=12)

# Run VQE calculation
params = np.random.randn(10)
coords = np.random.randn(10)
energy = quantum.vqe_circuit(params, coords)
```

## Benchmarking

```python
from benchmarks.membrane_protein_benchmark import MembraneProteinBenchmark

benchmark = MembraneProteinBenchmark()

# Evaluate on membrane proteins
results = benchmark.evaluate_membrane_proteins(model)
print(results)

# Evaluate on IDRs
idr_results = benchmark.evaluate_idrs(model)
print(idr_results)
```

## API Integration

```python
from qnhd_fold.experimental_apis import ExperimentalDataAPI

api = ExperimentalDataAPI()

# Fetch PDB structure
pdb_data = api.fetch_pdb_structure('1UBQ')
print(f"Resolution: {pdb_data['resolution']} Å")

# Fetch UniProt entry
uniprot_data = api.fetch_uniprot_entry('P0DTC2')
print(f"Sequence: {uniprot_data['sequence'][:50]}...")
```

## Visualization with PyMOL

```bash
# Load structure in PyMOL
pymol QNHD_predicted_structure.pdb

# Or use the provided script
@visualize_pymol.pml
```

## Tips for Best Results

1. **For short proteins (<100 residues)**: Use default parameters
2. **For large proteins (>500 residues)**: Increase num_diffusion_steps to 200
3. **For membrane proteins**: Results are already optimized
4. **For IDRs**: Expect lower pLDDT (50-70 is normal)

## Troubleshooting

**Q: Low confidence predictions?**
A: Try increasing num_diffusion_steps or check if protein has limited evolutionary signal

**Q: Out of memory?**
A: Reduce batch size or use a GPU with more memory

**Q: Quantum circuits slow?**
A: Use classical simulator for n_qubits <= 20

## Citation

If you use QNHD-Fold, please cite:
```
Marena, T., Rizk, D., & Shiraskar, S. (2026).
QNHD-Fold: Quantum-Neural Hybrid Diffusion for Protein Structure Prediction.
bioRxiv.
```
