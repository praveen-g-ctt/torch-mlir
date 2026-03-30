#!/usr/bin/env python3
"""
Simple test for MXFP8 type support - shows Torch IR generation only.
For full pipeline test including Linalg lowering, additional work is needed.
"""

import torch
import torch.nn as nn
from torch_mlir import fx


class SimpleMXFP8Test(nn.Module):
    """Simple module demonstrating MXFP8 scale type usage."""

    def forward(self, x, scale):
        # Convert scale to same dtype as x before multiplication
        return x * scale.to(x.dtype)


def main():
    print("=" * 70)
    print("MXFP8 Type Support Test for torch-mlir")
    print("=" * 70)
    print()

    # Check PyTorch support
    try:
        dtype = torch.float8_e8m0fnu
        print(f"✓ torch.float8_e8m0fnu is available: {dtype}")
    except AttributeError:
        print("✗ torch.float8_e8m0fnu not available in this PyTorch version")
        return
    print()

    # Create test module and inputs
    model = SimpleMXFP8Test()
    x = torch.randn(32, 32).to(torch.float8_e4m3fn)  # Element data
    scale = torch.randn(32, 1).to(torch.float8_e8m0fnu)  # Scale data

    print(f"Input tensor:  shape={x.shape}, dtype={x.dtype}")
    print(f"Scale tensor:  shape={scale.shape}, dtype={scale.dtype}")
    print()

    # Export and import to torch-mlir
    print("Exporting to torch-mlir...")
    mlir_module = fx.export_and_import(model, x, scale, func_name="mxfp8_test")
    print("✓ Successfully exported!")
    print()

    # Print the Torch dialect IR
    print("=" * 70)
    print("TORCH DIALECT IR")
    print("=" * 70)
    print(mlir_module)
    print()

    # Note about Linalg lowering
    print("=" * 70)
    print("LINALG LOWERING STATUS")
    print("=" * 70)
    print()
    print("⚠ Full Linalg lowering requires additional implementation:")
    print()
    print("Files to update:")
    print("  • lib/Conversion/Utils/Utils.cpp")
    print("    → Add Float8E8M0FNUType case to getTypeForScalarType()")
    print()
    print("  • lib/Conversion/TorchToLinalg/DataMovement.cpp")
    print("    → Ensure convert_element_type handles f8E8M0FNU conversions")
    print()
    print("The Torch dialect import is complete and working!")
    print("=" * 70)


if __name__ == "__main__":
    main()
