#!/usr/bin/env python3
"""
Test different operation patterns with MXFP8 types.

This demonstrates:
1. Direct operations on FP8 element data
2. Scale application (dequantization)
3. Mixed-precision operations
"""

import torch
import torch.nn as nn
from torch_mlir import fx


print("=" * 70)
print("MXFP8 OPERATION PATTERNS")
print("=" * 70)
print()

# Check PyTorch support
try:
    dtype_scale = torch.float8_e8m0fnu
    dtype_elem = torch.float8_e4m3fn
    print(f"✓ MXFP8 types available")
    print(f"  - Element type: {dtype_elem}")
    print(f"  - Scale type:   {dtype_scale}")
except AttributeError:
    print("✗ MXFP8 types not available")
    exit(1)
print()


# =============================================================================
# Pattern 1: Direct FP8 Operations (on element data only)
# =============================================================================
print("=" * 70)
print("Pattern 1: Direct FP8 Element Operations")
print("=" * 70)
print()
print("Operations directly on f8E4M3FN element data (no scaling)")
print()


class DirectFP8Operations(nn.Module):
    """Operations directly on FP8 element data."""

    def forward(self, x, y):
        # Both inputs are f8E4M3FN - no scaling involved
        # This represents operations on the quantized values directly
        return x + y


model1 = DirectFP8Operations()
x1 = torch.randn(8, 8).to(torch.float8_e4m3fn)
y1 = torch.randn(8, 8).to(torch.float8_e4m3fn)

try:
    mlir1 = fx.export_and_import(model1, x1, y1, func_name="direct_fp8_ops")
    print("✓ Direct FP8 operations export succeeded")
    print()
    print("Generated IR:")
    print("-" * 70)
    print(mlir1)
    print()
except Exception as e:
    print(f"✗ Export failed: {e}")
    print()


# =============================================================================
# Pattern 2: Scale Application (Dequantization)
# =============================================================================
print("=" * 70)
print("Pattern 2: MXFP8 with Scale Application (Dequantization)")
print("=" * 70)
print()
print("This is the typical MXFP8 pattern:")
print("  1. Element data (f8E4M3FN) stored in blocks")
print("  2. Scale factors (f8E8M0FNU) one per block")
print("  3. Dequantize: result = element_data * scale")
print()


class MXFP8Dequantize(nn.Module):
    """Dequantize MXFP8: apply scale to element data."""

    def forward(self, elements, scales):
        # elements: f8E4M3FN [batch, features]
        # scales: f8E8M0FNU [batch, 1] (one scale per row/block)

        # Convert scale to element dtype, then multiply (dequantize)
        scales_fp8 = scales.to(elements.dtype)
        dequantized = elements * scales_fp8
        return dequantized


model2 = MXFP8Dequantize()
elements = torch.randn(32, 32).to(torch.float8_e4m3fn)
scales = torch.randn(32, 1).to(torch.float8_e8m0fnu)

try:
    mlir2 = fx.export_and_import(model2, elements, scales, func_name="mxfp8_dequantize")
    print("✓ MXFP8 dequantization export succeeded")
    print()
    print("Generated IR:")
    print("-" * 70)
    print(mlir2)
    print()
except Exception as e:
    print(f"✗ Export failed: {e}")
    print()


# =============================================================================
# Pattern 3: Mixed Precision Workflow
# =============================================================================
print("=" * 70)
print("Pattern 3: Mixed-Precision Compute")
print("=" * 70)
print()
print("Typical workflow for MXFP8 computation:")
print("  1. Dequantize to higher precision (e.g., FP16/BF16)")
print("  2. Perform computation in higher precision")
print("  3. (Optionally) requantize back to MXFP8")
print()


class MixedPrecisionMXFP8(nn.Module):
    """Mixed-precision: dequantize, compute in FP32, requantize."""

    def forward(self, x_elem, x_scale, y_elem, y_scale):
        # Dequantize both inputs to FP32
        x = x_elem.to(torch.float32) * x_scale.to(torch.float32)
        y = y_elem.to(torch.float32) * y_scale.to(torch.float32)

        # Compute in FP32
        result = x @ y.T  # Matrix multiplication

        # For demonstration, return FP32 result
        # In practice, you'd requantize back to MXFP8
        return result


model3 = MixedPrecisionMXFP8()
x_elem = torch.randn(16, 32).to(torch.float8_e4m3fn)
x_scale = torch.randn(16, 1).to(torch.float8_e8m0fnu)
y_elem = torch.randn(16, 32).to(torch.float8_e4m3fn)
y_scale = torch.randn(16, 1).to(torch.float8_e8m0fnu)

try:
    mlir3 = fx.export_and_import(
        model3, x_elem, x_scale, y_elem, y_scale, func_name="mixed_precision_mxfp8"
    )
    print("✓ Mixed-precision export succeeded")
    print()
    print("Generated IR (truncated):")
    print("-" * 70)
    ir_lines = str(mlir3).split("\n")
    # Show first 40 lines
    print("\n".join(ir_lines[:40]))
    if len(ir_lines) > 40:
        print(f"... ({len(ir_lines) - 40} more lines)")
    print()
except Exception as e:
    print(f"✗ Export failed: {e}")
    print()


# =============================================================================
# Summary
# =============================================================================
print("=" * 70)
print("SUMMARY: MXFP8 Operation Patterns")
print("=" * 70)
print()
print("1. Direct FP8 Operations:")
print("   ✓ Can operate directly on f8E4M3FN element data")
print("   ✓ Fast, low memory")
print("   ✗ Limited precision (4 mantissa bits)")
print("   Use case: Simple operations where quantization error is acceptable")
print()
print("2. MXFP8 with Scaling (Dequantization):")
print("   ✓ Restores dynamic range via per-block scaling")
print("   ✓ Better accuracy than pure FP8")
print("   ~ Requires scale conversion/multiplication")
print("   Use case: Standard MXFP8 workflow, hardware accelerated on modern GPUs")
print()
print("3. Mixed-Precision:")
print("   ✓ Best accuracy - compute in FP16/FP32")
print("   ✓ Can requantize back to MXFP8 for storage/memory")
print("   ✗ Higher compute cost, more memory bandwidth")
print("   Use case: When accuracy is critical, MXFP8 used for storage only")
print()
print("Hardware Acceleration:")
print("  • NVIDIA Blackwell GPUs have native MXFP8 matrix multiply units")
print("  • These can operate on scaled blocks efficiently")
print("  • Software stacks may expose this as blocked GEMM operations")
print()
