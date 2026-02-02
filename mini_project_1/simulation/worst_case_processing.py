import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# User settings
# -----------------------------
FILENAME = "mini_project_1_worst_case_11_components_1k.txt"
MAKE_PLOTS = True

# -----------------------------
# Load and split runs
# -----------------------------
runs = []
current_time = []
current_v = []

with open(FILENAME) as f:
    for line in f:
        line = line.strip()

        # New run marker
        if line.startswith("Step Information"):
            if current_time:
                runs.append((np.array(current_time), np.array(current_v)))
                current_time = []
                current_v = []
            continue

        # Skip header / empty lines
        if line.startswith("time") or not line:
            continue

        parts = line.split()
        if len(parts) == 2:
            try:
                t, v = map(float, parts)
                current_time.append(t)
                current_v.append(v)
            except ValueError:
                pass

# Append last run
if current_time:
    runs.append((np.array(current_time), np.array(current_v)))

print(f"Loaded {len(runs)} runs")


# -----------------------------
# Frequency extraction function
# -----------------------------
def extract_frequency(time, v):
    vmin, vmax = np.min(v), np.max(v)

    # Reject non-oscillating runs
    if vmax - vmin < 1e-3:
        return None

    vth = 0.5 * (vmin + vmax)

    idx = np.where((v[:-1] < vth) & (v[1:] >= vth))[0]
    if len(idx) < 2:
        return None

    # Linear interpolation of threshold crossing
    t_cross = time[idx] + (vth - v[idx]) * (time[idx + 1] - time[idx]) / (
        v[idx + 1] - v[idx]
    )

    periods = np.diff(t_cross)
    freq = 1.0 / periods

    return freq


# -----------------------------
# Analyze all runs
# -----------------------------
results = []

for run_id, (time, v) in enumerate(runs):
    freq = extract_frequency(time, v)
    if freq is None:
        continue

    results.append(
        {
            "run": run_id,
            "mean": np.mean(freq),
            "min": np.min(freq),
            "max": np.max(freq),
            "std": np.std(freq),
            "cycles": len(freq),
        }
    )

print(f"Valid oscillating runs: {len(results)}")

# -----------------------------
# Worst-case analysis
# -----------------------------
slowest = min(results, key=lambda x: x["mean"])
fastest = max(results, key=lambda x: x["mean"])

global_min = min(r["min"] for r in results)
global_max = max(r["max"] for r in results)

print("\n--- WORST CASE RESULTS ---")
print(f"Slowest mean freq : {slowest['mean']:.3f} Hz (Run {slowest['run']})")
print(f"Fastest mean freq : {fastest['mean']:.3f} Hz (Run {fastest['run']})")
print(f"Absolute min freq : {global_min:.3f} Hz")
print(f"Absolute max freq : {global_max:.3f} Hz")

# -----------------------------
# Optional plots
# -----------------------------
if MAKE_PLOTS:
    means = [r["mean"] for r in results]

    plt.figure()
    plt.hist(means, bins=40)
    plt.xlabel("Mean Frequency (Hz)")
    plt.ylabel("Run Count")
    plt.title("Worst-Case Frequency Distribution")
    plt.grid(True)
    plt.show()

    # Plot example waveform + crossings from worst-case run
    t_wc, v_wc = runs[slowest["run"]]
    freq_wc = extract_frequency(t_wc, v_wc)
    vth_wc = 0.5 * (np.min(v_wc) + np.max(v_wc))

    plt.figure()
    plt.plot(t_wc, v_wc)
    plt.axhline(vth_wc, linestyle="--")
    plt.title(f"Worst-case waveform (Run {slowest['run']})")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.grid(True)
    plt.show()
