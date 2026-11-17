## SKEW_TILED_BF16 — Verified NVIDIA A40 (`sm_86`) Receipts & Unified Logic Protocol

Full tables, raw logs, and multi-model red-team protocol live at **[cwtractors.com](https://cwtractors.com)**

- **Architecture:** NVIDIA A40 (Ampere `sm_86`, 48 GB GDDR6, 300 W TDP)
- **Register Footprint:** 15 regs/thread (`bf16_t`, 0 spills) vs. 36 regs/thread (`float`)
- **Correctness:** `0` site disagreements across all 6 sub-lattice phases against both FP32 parity and `FP64 std::exp` reference (`L = 384, 768`, random &plusmn;1 and uniform `+1` ensembles)
- **Throughput (30 interleaved trials):** **10.204 ms** vs. 15.291 ms at `L = 4608` (**1.499x speedup**); **40.679 ms** vs. 60.064 ms at `L = 9216` (**1.477x speedup**)
- **NVML Dynamic Energy (16 paired ABBA quads, `P_idle = 73.02 W`):** **186.735** vs. 261.164 mJ/sweep at `L = 4608` (**28.5% reduction**); **788.288** vs. 1083.034 mJ/sweep at `L = 9216` (**27.1% reduction**)
- **Raw Receipts:** [`a40_environment_receipt.txt`](./a40_environment_receipt.txt) | [`a40_nsys_summary.txt`](./a40_nsys_summary.txt)
- **PMC Disclosure:** Hardware performance counters (`ncu`) were restricted by the host container (`ERR_NVGPUCTRPERM`); DRAM byte reduction is inferred from 2B vs. 4B representation and corroborated by wall-clock speedup and NVML power logs.# a40-gemv-skewed-tiling
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
