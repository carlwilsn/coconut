#!/usr/bin/env python3
"""Session A driver — ProsQA (CoT bf16, w/o-thought fp32, Coconut fp32).

Runs on the Lambda A10, launched DETACHED via `lambda run_bg` (tmux) so it
outlives the SSH session. For each run it:
  1. trains (torchrun) to the config's num_epochs, teeing to logs/<name>_train.log;
  2. parses the best VALIDATION accuracy line and maps it to a checkpoint
     (checkpoint_{k+1} for the 0-indexed k-th "Accuracy on validation set" line,
     since resume=0 for every Session A config — verified against run.py);
  3. writes an eval config (only_eval:True, load_model_path=best ckpt,
     val_path=<test set>, resume high enough to sit in the final fully-latent
     curriculum stage) and evals that checkpoint on the TEST set (paper-comparable);
  4. copies the ONE best checkpoint to best_ckpts/<name>/ + writes META.json
     (val acc, test acc, epoch, ckpt name, size);
  5. touches RUN_<name>_DONE.txt.
After all three: writes SESSION_A_SUMMARY.json and touches SESSION_A_DONE.txt.

DESIGN NOTES (why it's shaped this way):
- torchrun's exit code is IGNORED on purpose. A run can end by (a) finishing all
  epochs or (b) me `pkill`-ing it for a documented early-cut once val is flat near
  target. Both leave the per-epoch checkpoints on disk (save_only_improve handles
  CoT; the latent runs save every epoch), so "parse whatever's in the log + eval the
  best" is correct in both cases. This is what makes early-cut a clean `pkill`.
- A per-run wall-clock backstop (PER_RUN_MAX_HOURS) kills a runaway so an
  unattended run can never bill forever.
- Completion is signalled by SENTINEL FILES only (never live progress), per the
  2026-06-19 idle-bill lesson. The box does NOT self-terminate via the Lambda API
  (the API key is not exposed to rented boxes); teardown is enforced from the
  always-on homestation by WatchRun's terminate_command + supervisor wakes.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time

HOME = os.path.expanduser("~")
REPO = os.path.join(HOME, "coconut")
os.chdir(REPO)
os.makedirs("logs", exist_ok=True)
os.makedirs("best_ckpts", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

ENV = dict(os.environ)
ENV["WANDB_MODE"] = "disabled"
ENV["TF_CPP_MIN_LOG_LEVEL"] = "3"

PER_RUN_MAX_HOURS = 16.0  # wall-clock backstop per run (ProsQA fp32 should be well under this)

TEST_PATH = "data/prosqa_test.json"

# Order: cheapest/most-diagnostic first. mode drives the eval-config curriculum stage.
RUNS = [
    {
        "name": "prosqa_cot",
        "cfg": "args_a10/prosqa_cot_a10.yaml",
        "mode": "cot",          # scheduled_stage is always 0; no latent thoughts at eval
    },
    {
        "name": "prosqa_nothought",
        "cfg": "args_a10/prosqa_nothought_a10_fp32.yaml",
        "mode": "latent",       # eval in final fully-latent stage (resume high)
    },
    {
        "name": "prosqa_coconut",
        "cfg": "args_a10/prosqa_coconut_a10_fp32.yaml",
        "mode": "latent",
    },
]


def read_yaml_flat(path):
    """Minimal flat-YAML reader preserving raw value text. Configs are flat key: value."""
    d = {}
    order = []
    for line in open(path):
        s = line.rstrip("\n")
        if not s.strip() or s.lstrip().startswith("#"):
            continue
        if ":" not in s:
            continue
        k, v = s.split(":", 1)
        k = k.strip()
        d[k] = v.strip()
        order.append(k)
    return d, order


def save_dir_for(cfg_path):
    d, _ = read_yaml_flat(cfg_path)
    return os.path.join(d["save_path"], d["name"])  # run.py: os.path.join(save_path, name)


def parse_val_accs(log_path):
    """Return list of (epoch_0idx, frac) from 'Accuracy on validation set: c / t = frac'."""
    accs = []
    if not os.path.exists(log_path):
        return accs
    pat = re.compile(r"Accuracy on validation set:\s+(\d+)\s+/\s+(\d+)\s+=\s+([\d.]+)")
    for line in open(log_path, errors="replace"):
        m = pat.search(line)
        if m:
            accs.append(float(m.group(3)))
    return accs


def run_torchrun(cfg_path, log_path, max_hours):
    """Launch torchrun; stream to log; enforce wall-clock backstop. Exit code ignored."""
    cmd = ["torchrun", "--nnodes", "1", "--nproc_per_node", "1", "run.py", cfg_path]
    with open(log_path, "w") as lf:
        lf.write(f"# driver: launching {' '.join(cmd)} at {time.ctime()}\n")
        lf.flush()
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=ENV)
        deadline = time.time() + max_hours * 3600
        while p.poll() is None:
            if time.time() > deadline:
                lf.write(f"\n# driver: PER_RUN_MAX_HOURS ({max_hours}h) hit — killing.\n")
                lf.flush()
                p.terminate()
                try:
                    p.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    p.kill()
                break
            time.sleep(20)
    return


def write_eval_cfg(train_cfg, out_cfg, best_ckpt_abs, mode):
    d, order = read_yaml_flat(train_cfg)
    over = {
        "only_eval": "True",
        "load_model_path": best_ckpt_abs,
        "val_path": TEST_PATH,
        "name": d["name"] + "-eval",
        # For latent runs, resume must place us in the final fully-latent stage
        # (dataset.py clamps scheduled_stage to max_latent_stage). CoT ignores it.
        "resume": "40" if mode == "latent" else "0",
    }
    d.update(over)
    with open(out_cfg, "w") as f:
        f.write(f"# auto-generated eval config for {d['name']} (test-set eval)\n")
        for k in order:
            f.write(f"{k}: {d[k]}\n")
        for k in over:
            if k not in order:
                f.write(f"{k}: {d[k]}\n")


def do_run(run):
    name, cfg, mode = run["name"], run["cfg"], run["mode"]
    train_log = f"logs/{name}_train.log"
    eval_log = f"logs/{name}_eval.log"
    print(f"=== [{name}] TRAIN START {time.ctime()} ===", flush=True)
    run_torchrun(cfg, train_log, PER_RUN_MAX_HOURS)
    print(f"=== [{name}] TRAIN ENDED {time.ctime()} ===", flush=True)

    accs = parse_val_accs(train_log)
    meta = {"name": name, "cfg": cfg, "mode": mode, "val_trajectory": accs}
    if not accs:
        meta["status"] = "ERROR_no_val_lines"
        json.dump(meta, open(f"best_ckpts/{name}_META.json", "w"), indent=2)
        open(f"RUN_{name}_DONE.txt", "w").write("ERROR: no validation lines parsed\n")
        print(f"=== [{name}] ERROR no val lines ===", flush=True)
        return meta

    best_k = max(range(len(accs)), key=lambda i: accs[i])
    best_val = accs[best_k]
    best_ckpt = f"checkpoint_{best_k + 1}"
    sdir = save_dir_for(cfg)
    best_ckpt_abs = os.path.abspath(os.path.join(sdir, best_ckpt))
    meta.update({"best_epoch_0idx": best_k, "best_ckpt": best_ckpt,
                 "best_val_acc": best_val, "n_epochs_seen": len(accs),
                 "checkpoint_dir": sdir})
    print(f"=== [{name}] best val {best_val:.4f} @epoch{best_k+1} -> {best_ckpt} ===", flush=True)

    # Test-set eval of the best checkpoint
    test_acc = None
    if os.path.exists(best_ckpt_abs):
        eval_cfg = f"args_a10/{name}_eval_gen.yaml"
        write_eval_cfg(cfg, eval_cfg, best_ckpt_abs, mode)
        print(f"=== [{name}] TEST EVAL START {time.ctime()} ===", flush=True)
        run_torchrun(eval_cfg, eval_log, 3.0)
        eaccs = parse_val_accs(eval_log)
        test_acc = eaccs[-1] if eaccs else None
        meta["test_acc"] = test_acc
        print(f"=== [{name}] TEST acc = {test_acc} ===", flush=True)
        # Copy the best checkpoint back (single file)
        dst_dir = os.path.join("best_ckpts", name)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, best_ckpt)
        try:
            shutil.copy2(best_ckpt_abs, dst)
            meta["saved_ckpt_path"] = os.path.abspath(dst)
            meta["saved_ckpt_bytes"] = os.path.getsize(dst)
        except Exception as e:  # noqa
            meta["copy_error"] = str(e)
    else:
        meta["status"] = "ERROR_best_ckpt_missing"
        meta["expected_ckpt"] = best_ckpt_abs

    meta.setdefault("status", "ok")
    json.dump(meta, open(f"best_ckpts/{name}_META.json", "w"), indent=2)
    open(f"RUN_{name}_DONE.txt", "w").write(
        f"{name}: best_val={best_val:.4f}@ep{best_k+1} test_acc={test_acc} ckpt={best_ckpt}\n")
    return meta


def main():
    summary = {"session": "A", "started": time.ctime(), "runs": []}
    for run in RUNS:
        try:
            summary["runs"].append(do_run(run))
        except Exception as e:  # noqa
            summary["runs"].append({"name": run["name"], "status": f"EXCEPTION: {e}"})
            open(f"RUN_{run['name']}_DONE.txt", "w").write(f"EXCEPTION: {e}\n")
        json.dump(summary, open("SESSION_A_SUMMARY.json", "w"), indent=2)
    summary["finished"] = time.ctime()
    json.dump(summary, open("SESSION_A_SUMMARY.json", "w"), indent=2)
    open("SESSION_A_DONE.txt", "w").write("session A complete\n" + json.dumps(summary, indent=2))
    print(f"=== SESSION A DONE {time.ctime()} ===", flush=True)


if __name__ == "__main__":
    main()
