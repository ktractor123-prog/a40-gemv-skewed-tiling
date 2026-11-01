import sys
import math
import platform
import statistics
import subprocess
import torch
from naive_gemv import load_naive
from naive_cond import load_naive_cond
from naive_cond_16x16 import load_naive_cond_16x16
from skew_tiled import load_skew_tiled

KNOWN_GPU_BW = {
    "NVIDIA A40": 696.0,
    "NVIDIA A100": 1555.0,
    "NVIDIA H100": 3350.0,
    "NVIDIA L40": 864.0,
    "NVIDIA RTX 4090": 1008.0,
    "NVIDIA RTX 3090": 936.0,
}

def compute_ref(x_host, L, pitch, d4):
    x64 = x_host.to(torch.float64)
    y_ref = torch.zeros((L, pitch), dtype=torch.float64)

    phys = torch.zeros((L, L), dtype=torch.float64)
    for q in range(L):
        for r in range(L):
            phys[q, r] = x64[((q + r) % L + L) % L, r]

    acc = torch.zeros((L, L), dtype=torch.float64)
    for a, b in d4:
        acc += torch.roll(torch.roll(phys, shifts=a, dims=0), shifts=b, dims=1)

    for q in range(L):
        for r in range(L):
            y_ref[((q + r) % L + L) % L, r] = acc[q, r]

    return y_ref

def calc_p95(sorted_times):
    if not sorted_times:
        return 0.0
    k = (len(sorted_times) - 1) * 0.95
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_times[int(k)]
    return sorted_times[int(f)] * (c - k) + sorted_times[int(c)] * (k - f)

def time_batched(launch_fn, reps=20, warmup=3, iters=20):
    for _ in range(warmup):
        launch_fn(1)
    torch.cuda.synchronize()

    times = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(iters):
        start_event.record()
        launch_fn(reps)
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event) / reps)

    torch.cuda.synchronize()
    times.sort()
    med = statistics.median(times)
    p95_val = calc_p95(times)
    lo = times[0]
    hi = times[-1]
    return med, p95_val, lo, hi

def triple(fn, reps):
    meds, p95s, mins, maxs = [], [], [], []
    for _ in range(3):
        m, p, lo, hi = time_batched(fn, reps=reps, warmup=3, iters=20)
        meds.append(m); p95s.append(p); mins.append(lo); maxs.append(hi)
    meds.sort(); p95s.sort(); mins.sort(); maxs.sort()
    return meds[1], p95s[1], mins[0], maxs[-1], meds

def get_theoretical_bandwidth(props):
    # 1. Try PyTorch device properties
    clock_khz = getattr(props, "memory_clock_rate", 0)
    bus_bits = getattr(props, "memory_bus_width", 0)
    
    if clock_khz > 0 and bus_bits > 0:
        return 2.0 * (clock_khz * 1e3) * (bus_bits / 8.0) / 1e9

    # 2. Try nvidia-smi parsing
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.max.memory,bus_width", "--format=csv,noheader,nounits"]
        ).decode("utf-8").strip()
        parts = [p.strip() for p in out.split(",")]
        if len(parts) >= 2 and parts[0].replace('.','',1).isdigit() and parts[1].isdigit():
            m_clock_mhz = float(parts[0])
            m_bus_bits = float(parts[1])
            if m_clock_mhz > 0 and m_bus_bits > 0:
                return 2.0 * (m_clock_mhz * 1e6) * (m_bus_bits / 8.0) / 1e9
    except Exception:
        pass

    # 3. Fallback to Known Hardware Lookup
    gpu_name = torch.cuda.get_device_name(0)
    for key, bw in KNOWN_GPU_BW.items():
        if key.lower() in gpu_name.lower():
            return bw

    return 0.0

def main():
    if not torch.cuda.is_available():
        print("BLOCKER: CUDA is not available.")
        sys.exit(1)

    torch.manual_seed(0)

    # 1. HEADER BLOCK
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Python Version: {platform.python_version()}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Version: {torch.version.cuda}")
    try:
        driver_ver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        ).decode("utf-8").strip()
    except Exception:
        driver_ver = "Unknown"
    print(f"Driver Version: {driver_ver}")

    l2_bytes = getattr(props, "L2_cache_size", getattr(props, "l2_cache_size", 0))
    print(f"L2 Cache: {l2_bytes / (1024*1024):.1f} MB")

    # 3. THEORETICAL BANDWIDTH CALCULATION & STRICT ZERO-GUARD
    theoretical_bw = get_theoretical_bandwidth(props)
    if theoretical_bw <= 0.0:
        print("BLOCKER: Theoretical memory bandwidth calculation yielded <= 0.0 GB/s. Driver/device query failed.")
        sys.exit(1)
    
    print(f"Theoretical Bandwidth: {theoretical_bw:.1f} GB/s")
    print()

    naive_mod = load_naive()
    naive_cond = load_naive_cond()
    naive_cond_16 = load_naive_cond_16x16()
    skew_tiled = load_skew_tiled()

    d4_tuples = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    d4_flat = []
    for a, b in d4_tuples:
        d4_flat.extend([a, b])
    d4_tensor = torch.tensor(d4_flat, dtype=torch.int32, device="cuda")

    l_list = [16, 32, 48, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    large_sizes = {2048, 4096, 8192}
    sentinel_val = -12345.0
    reps = 20

    print("WARNING: Logical bytes != DRAM transactions. Achieved GB/s > theoretical DRAM ceiling indicates coalescing reuse. The theoretical ceiling applies to real traffic.")
    print()

    print("Quattra Tiled Benchmark (3 rounds of 20 iters per L, median & p95 reported)")
    header = (
        f"{'L':<6}{'N':<10}"
        f"{'Mod_med':<11}{'Mod_p95':<11}"
        f"{'Cond_med':<11}{'Cond_p95':<11}"
        f"{'C16_med':<11}{'C16_p95':<11}"
        f"{'Tile_med':<11}{'Tile_p95':<11}"
        f"{'Speedup':<10}"
        f"{'Naive GB/s':<13}{'Tiled GB/s':<13}"
        f"{'Max Abs Diff':<13}"
    )
    print(header)
    print("-" * len(header))

    pass_criteria_2_wins = 0
    rule3_audit_log = []

    for L in l_list:
        N = L * L
        pitch = 32 * math.ceil(L / 32)

        x = torch.randn((L, pitch), dtype=torch.float32, device="cuda")
        if pitch > L:
            x[:, L:pitch] = float("nan")

        y_mod = torch.full((L, pitch), sentinel_val, dtype=torch.float32, device="cuda")
        y_cond = torch.full((L, pitch), sentinel_val, dtype=torch.float32, device="cuda")
        y_cond16 = torch.full((L, pitch), sentinel_val, dtype=torch.float32, device="cuda")
        y_tiled = torch.full((L, pitch), sentinel_val, dtype=torch.float32, device="cuda")

        naive_mod.launch_naive_gemv(x, y_mod, d4_tensor, L, pitch, 1)
        naive_cond.launch_naive_cond(x, y_cond, d4_tensor, L, pitch, 1)
        naive_cond_16.launch_naive_cond_16x16(x, y_cond16, d4_tensor, L, pitch, 1)
        skew_tiled.launch_skew_tiled(x, y_tiled, L, pitch, 1)
        torch.cuda.synchronize()

        if pitch > L:
            for name, y_buf in [("NaiveMod", y_mod), ("NaiveCond", y_cond), ("Cond16x16", y_cond16), ("SkewTiled", y_tiled)]:
                pad = y_buf[:, L:pitch]
                if not torch.all(pad == sentinel_val):
                    print(f"BLOCKER: Output padding sentinel corrupted in {name} at L={L}")
                    sys.exit(1)

        for name, y_buf in [("NaiveMod", y_mod), ("NaiveCond", y_cond), ("Cond16x16", y_cond16), ("SkewTiled", y_tiled)]:
            valid_area = y_buf[:, :L]
            if torch.isnan(valid_area).any():
                print(f"BLOCKER: NaN detected in valid domain of {name} at L={L}")
                sys.exit(1)

        x_cpu = x.cpu()
        y_ref = compute_ref(x_cpu, L, pitch, d4_tuples)
        y_ref_cuda = y_ref.to(device="cuda", dtype=torch.float32)

        max_diff_mod = torch.max(torch.abs(y_mod[:, :L] - y_ref_cuda[:, :L])).item()
        max_diff_cond = torch.max(torch.abs(y_cond[:, :L] - y_ref_cuda[:, :L])).item()
        max_diff_cond16 = torch.max(torch.abs(y_cond16[:, :L] - y_ref_cuda[:, :L])).item()
        max_diff_tiled = torch.max(torch.abs(y_tiled[:, :L] - y_ref_cuda[:, :L])).item()

        max_abs_diff = max(max_diff_mod, max_diff_cond, max_diff_cond16, max_diff_tiled)
        if max_abs_diff > 1e-5:
            print(f"BLOCKER: Absolute tolerance 1e-5 exceeded at L={L}, max_abs_diff={max_abs_diff}")
            sys.exit(1)

        mod_med, mod_p95, mod_min, mod_max, mod_rounds = triple(
            lambda r: naive_mod.launch_naive_gemv(x, y_mod, d4_tensor, L, pitch, r), reps)
        cond_med, cond_p95, cond_min, cond_max, cond_rounds = triple(
            lambda r: naive_cond.launch_naive_cond(x, y_cond, d4_tensor, L, pitch, r), reps)
        cond16_med, cond16_p95, cond16_min, cond16_max, cond16_rounds = triple(
            lambda r: naive_cond_16.launch_naive_cond_16x16(x, y_cond16, d4_tensor, L, pitch, r), reps)
        tiled_med, tiled_p95, tiled_min, tiled_max, tiled_rounds = triple(
            lambda r: skew_tiled.launch_skew_tiled(x, y_tiled, L, pitch, r), reps)

        best_control_med = min(cond_med, cond16_med)
        best_control_min = min(cond_min, cond16_min)
        speedup_vs_best = best_control_med / tiled_med if tiled_med > 0 else 0.0

        bytes_transferred = 4.0 * N * 5.0
        naive_gbs = (bytes_transferred / (best_control_med * 1e-3)) / 1e9 if best_control_med > 0 else 0.0
        tiled_gbs = (bytes_transferred / (tiled_med * 1e-3)) / 1e9 if tiled_med > 0 else 0.0

        print(
            f"{L:<6}{N:<10}"
            f"{mod_med:<11.5f}{mod_p95:<11.5f}"
            f"{cond_med:<11.5f}{cond_p95:<11.5f}"
            f"{cond16_med:<11.5f}{cond16_p95:<11.5f}"
            f"{tiled_med:<11.5f}{tiled_p95:<11.5f}"
            f"{speedup_vs_best:<10.3f}"
            f"{naive_gbs:<13.2f}{tiled_gbs:<13.2f}"
            f"{max_abs_diff:<13.2e}"
        )

        if L in large_sizes:
            is_3pct_faster = speedup_vs_best >= 1.03
            tiled_slowest_beats_control_fastest = tiled_max < best_control_min
            pass_rule3 = tiled_slowest_beats_control_fastest
            if is_3pct_faster and pass_rule3:
                pass_criteria_2_wins += 1
            
            rule3_audit_log.append({
                "L": L,
                "tiled_rounds": tiled_rounds,
                "tiled_max": tiled_max,
                "best_control_min": best_control_min,
                "pass_rule3": pass_rule3,
                "speedup": speedup_vs_best
            })

    print("\nRule 3 Distribution Audit (Per-Round Separation Verification):")
    print(f"{'L':<6}{'Tiled Rounds (ms)':<32}{'Slowest Tiled':<15}{'Fastest Control':<17}{'Separation Pass':<15}")
    print("-" * 85)
    for entry in rule3_audit_log:
        rounds_str = f"[{entry['tiled_rounds'][0]:.5f}, {entry['tiled_rounds'][1]:.5f}, {entry['tiled_rounds'][2]:.5f}]"
        pass_str = "PASS" if entry['pass_rule3'] else "FAIL"
        print(f"{entry['L']:<6}{rounds_str:<32}{entry['tiled_max']:<15.5f}{entry['best_control_min']:<17.5f}{pass_str:<15}")

    print("\nPre-Registered Pass/Fail Evaluation:")
    print("Pre-Registered Rules: (1) >3% speedup vs best control, (2) wins at >=2 of {2048, 4096, 8192}, (3) slowest-of-3 tiled < fastest-of-3 control.")
    if pass_criteria_2_wins >= 2:
        print(f"RESULT: PASS — skew_tiled achieved >= 3% speedup with statistically separating bounds at {pass_criteria_2_wins}/3 large sizes.")
    else:
        print(f"RESULT: NULL / FAIL — Large size wins meeting all criteria: {pass_criteria_2_wins}/3 (requires >= 2).")

if __name__ == "__main__":
    main()