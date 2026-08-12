import sys
from llamatune.hardware import get_hardware_profile
from llamatune.sweep import run_sweep, best_generation_config
from llamatune.serve import launch_server

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m llamatune.cli <path_to_gguf>")
        sys.exit(1)

    model_path = sys.argv[1]

    print("Detecting hardware...")
    hw = get_hardware_profile()
    print(hw)

    print("\nRunning benchmark sweep (this takes a minute)...")
    results = run_sweep(model_path, ngl_values=[0, 10, 20, 30, 50, 99])
    for r in results:
        print(f"  ngl={r.ngl}: {r.tokens_per_sec:.2f} tok/s")

    best = best_generation_config(results)
    if best is None:
        print("Sweep failed, no valid config found.")
        sys.exit(1)

    print(f"\nBest config: ngl={best.ngl} ({best.tokens_per_sec:.2f} tok/s)")
    print("Launching server with this config...\n")
    launch_server(model_path, ngl=best.ngl)

if __name__ == "__main__":
    main()