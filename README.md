# a40-gemv-skewed-tiling
CUDA gemV skewed-tiling kernel
# NVIDIA A40 GemV Skewed-Tiling Kernel Audit

Executable CUDA benchmark code and verification outputs demonstrating a **~11.9% latency reduction at sequence length L=8192** for GemV operations on NVIDIA A40 hardware.

## Optimization & Results
- **Hardware Target:** NVIDIA A40 (48GB GDDR6, PCIe)
- **Performance:** ~1.05x speedup at L=2048, scaling to **1.119x (11.9%) speedup at L=8192** over optimal baseline control.
- **Validation:** Verified across 60-sample extreme-bound runs with full statistical separation.

## Live Audit Receipts & Raw Logs
Full audit archives (`a40_gemv_audit_logs.tar.gz`), raw execution logs (`run1.txt` - `run3.txt`), and methodology notes (`AMENDMENT.txt`) are maintained at:
**[cwtractors.com](https://cwtractors.com)**

## Usage
Run local CUDA validation:
```bash
python3 validate_tiled.py
