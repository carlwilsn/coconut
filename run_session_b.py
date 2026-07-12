#!/usr/bin/env python3
"""Session B driver — ProntoQA (CoT bf16, No-CoT bf16, Coconut fp32).

Direct clone of run_session_a.py's proven logic, retargeted at the three ProntoQA
cells of Coconut Table 1. Launched DETACHED via `lambda run_bg` (tmux) so it
outlives the SSH session. For each run it:
  1. trains (torchrun) to the config's num_epochs, teeing to logs/<name>_train.log;
  2. parses the best VALIDATION accuracy line -> checkpoint_{k+1} (resume=0 for every
     Session B config, verified against run.py);
  3. writes an eval config (only_eval:True, load_model_path=best ckpt,
     val_path=data/prontoqa_test.json, resume high enough for the final fully-latent
     stage) and evals that checkpoint on the TEST set (paper-comparable);
  4. copies the ONE best checkpoint to best_ckpts/<name>/ + META.json;
  5. touches RUN_<name>_DONE.txt.
After all three: writes SESSION_B_SUMMARY.json and touches SESSION_B_DONE.txt.

DESIGN NOTES (identical rationale to Session A):
- torchrun's exit code is IGNORED on purpose (finish-all-epochs OR documented
  early-cut pkill both leave per-epoch checkpoints on disk; parse-log-then-eval-best
  is correct either way).
- PER_RUN_MAX_HOURS backstop kills a runaway. ProntoQA runs are SHORT (short
  sequences, 9k train examples) so 8h is generous; the coconut latent run is the
  slowest but still far under.  ***Lesson carried from Session A: the 16h backstop
  truncated the ProsQA coconut run mid-climb. ProntoQA's asymptote lands MUCH
  earlier (paper 99.8 by the final curriculum stage) and the dataset is 9k not 17.9k,
  so 8h is safe — but if a run is still climbing near the cap, DO NOT rely on the
  backstop; extend it.***
- Completion signalled by SENTINEL FILES only (never live progress).
- The box CANNOT self-terminate via the Lambda API; teardown is enforced from the
  homestation (always `lambda list` + terminate by hand; never trust a watcher's GONE).
"""
import json
import os
import re
import shutil
import subprocess
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

PER_RUN_MAX_HOURS = 8.0  # ProntoQA runs are short; generous backstop

TEST_PATH = "data/prontoqa_test.json"

# Order: cheapest/most-diagnostic baselines first, latent run last.
RUNS = [
    {"name": "prontoqa_cot", "cfg": "args_a10/prontoqa_cot_a10.yaml", "mode": "cot"},
    {"name": "prontoqa_nocot", "cfg": "args_a10/prontoqa_nocot_a10.yaml", "mode": "cot"},
    {"name": "prontoqa_coconut", "cfg": "args_a10/prontoqa_coconut_a10_fp32.yaml", "mode": "latent"},
]


def read_yaml_flat(path):
    d, order = {}, []
    for line in open(path):
        s = line.rstrip("\n")
        if not s.strip() or s.lstrip().startswith("#") or ":" not in s:
            continue
        k, v = s.split(":", 1)
        d[k.strip()] = v.strip()
        order.append(k.strip())
    return d, order


def save_dir_for(cfg_path):
    d, _ = read_yaml_flat(cfg_path)
    return os.path.join(d["save_path"], d["name"])


def parse_val_accs(log_path):
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


def write_eval_cfg(train_cfg, out_cfg, best_ckpt_abs, mode):
    d, order = read_yaml_flat(train_cfg)
    over = {
        "only_eval": "True",
        "load_model_path": best_ckpt_abs,
        "val_path": TEST_PATH,
        "name": d["name"] + "-eval",
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

    test_acc = None
    if os.path.exists(best_ckpt_abs):
        eval_cfg = f"args_a10/{name}_eval_gen.yaml"
        write_eval_cfg(cfg, eval_cfg, best_ckpt_abs, mode)
        print(f"=== [{name}] TEST EVAL START {time.ctime()} ===", flush=True)
        run_torchrun(eval_cfg, eval_log, 2.0)
        eaccs = parse_val_accs(eval_log)
        test_acc = eaccs[-1] if eaccs else None
        meta["test_acc"] = test_acc
        print(f"=== [{name}] TEST acc = {test_acc} ===", flush=True)
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
    summary = {"session": "B", "started": time.ctime(), "runs": []}
    for run in RUNS:
        try:
            summary["runs"].append(do_run(run))
        except Exception as e:  # noqa
            summary["runs"].append({"name": run["name"], "status": f"EXCEPTION: {e}"})
            open(f"RUN_{run['name']}_DONE.txt", "w").write(f"EXCEPTION: {e}\n")
        json.dump(summary, open("SESSION_B_SUMMARY.json", "w"), indent=2)
    summary["finished"] = time.ctime()
    json.dump(summary, open("SESSION_B_SUMMARY.json", "w"), indent=2)
    open("SESSION_B_DONE.txt", "w").write("session B complete\n" + json.dumps(summary, indent=2))
    print(f"=== SESSION B DONE {time.ctime()} ===", flush=True)


if __name__ == "__main__":
    main()
