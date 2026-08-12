import subprocess
import json
from dataclasses import dataclass

LLAMA_BENCH = "bin/llama-bench.exe"

@dataclass
class BenchResult:
    ngl: int
    is_generation: bool
    tokens_per_sec: float

def run_sweep(model_path: str, ngl_values: list[int]) -> list[BenchResult]:
    ngl_arg = ",".join(str(n) for n in ngl_values)
    # -p 0 skips prompt-processing tests, -n 128 runs only generation tests
    cmd = [LLAMA_BENCH, "-m", model_path, "-ngl", ngl_arg, "-p", "0", "-n", "128", "-o", "json"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("Sweep failed:", result.stderr)
        return []

    data = json.loads(result.stdout)
    results = []
    for entry in data:
        results.append(BenchResult(
            ngl=entry["n_gpu_layers"],
            is_generation=entry["n_gen"] > 0,
            tokens_per_sec=entry["avg_ts"],
        ))
    return results

def best_generation_config(results: list[BenchResult]) -> BenchResult | None:
    gen_results = [r for r in results if r.is_generation]
    if not gen_results:
        return None
    return max(gen_results, key=lambda r: r.tokens_per_sec)

if __name__ == "__main__":
    model_path = r"C:\Users\Debian\.cache\huggingface\hub\models--Qwen--Qwen2.5-Coder-1.5B-Instruct-GGUF\snapshots\f86cb2c1fa58255f8052cc32aeede1b7482d4361\qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
    results = run_sweep(model_path, ngl_values=[0, 10, 20, 99])
    for r in results:
        print(f"ngl={r.ngl}: {r.tokens_per_sec:.2f} tok/s")
    best = best_generation_config(results)
    print(f"\nBEST: ngl={best.ngl} -> {best.tokens_per_sec:.2f} tok/s")