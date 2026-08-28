# -*- coding: utf-8 -*-
"""Multi-seed IDC training campaign (GPU).

Runs run_idc.py for every (dataset, seed) pair as a SUBPROCESS (fresh CUDA
context per run), small datasets first. Skips pairs whose idc_out_<ds>_seed<k>.npz
already exists, so the campaign is resumable. 5 seeds for every dataset; MNIST
uses IDC's validated config with 700 epochs and is by far the slowest (~15 min
per run).

After this, run evaluate.py (metrics for both methods) and gen_results_table.py.
"""
import os, sys, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, "venv", "Scripts", "python.exe")

RUNS = [(ds, s) for ds in ["two_moons", "blobs", "iris", "breast_cancer",
                           "digits", "har", "cifar10", "mnist"] for s in range(5)]


def main():
    t0 = time.time()
    done, failed = [], []
    for i, (ds, seed) in enumerate(RUNS, 1):
        from wpaths import idc_out
        out = idc_out(f"idc_out_{ds}_seed{seed}.npz")
        tag = f"[{i}/{len(RUNS)}] {ds} seed={seed}"
        if os.path.exists(out):
            print(f"{tag}: exists, skipped", flush=True)
            done.append((ds, seed, "cached"))
            continue
        t = time.time()
        r = subprocess.run([PY, os.path.join(HERE, "run_idc.py"),
                            "--data", ds, "--seed", str(seed)],
                           capture_output=True, text=True, cwd=HERE)
        dt = time.time() - t
        if r.returncode == 0 and os.path.exists(out):
            print(f"{tag}: OK ({dt/60:.1f} min)", flush=True)
            done.append((ds, seed, f"{dt/60:.1f}min"))
        else:
            print(f"{tag}: FAILED rc={r.returncode} ({dt/60:.1f} min)", flush=True)
            tail = "\n".join((r.stdout + "\n" + r.stderr).splitlines()[-15:])
            print(tail, flush=True)
            failed.append((ds, seed))
    print(f"\nCAMPAIGN DONE in {(time.time()-t0)/3600:.2f} h: "
          f"{len(done)} ok, {len(failed)} failed: {failed}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
