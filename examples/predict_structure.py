#!/usr/bin/env python3
"""Example: Predict protein structure with QNHD-Fold"""

from qnhd_fold import QNHDFold

# Initialize model
model = QNHDFold()

# Your protein sequence
sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNL"

# Predict structure
print(f"Predicting structure for {len(sequence)}-residue protein...")
structure = model.predict_structure(sequence, num_diffusion_steps=100)

# Results
print(f"\nResults:")
print(f"  Mean confidence (pLDDT): {structure.confidence.mean():.1f}")
print(f"  High confidence residues: {(structure.confidence > 90).sum()}/{len(sequence)}")

# Save to PDB
structure.save("predicted_structure.pdb")
print(f"\n✓ Structure saved to predicted_structure.pdb")
print("  View with: pymol predicted_structure.pdb")
