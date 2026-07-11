#!/usr/bin/env python3
"""Continuation driver — resume ProsQA Coconut fp32 to the ep30 fully-latent asymptote.

The original Session-A coconut run was cost-truncated at epoch 19 by the 16h
per-run backstop while validation was still climbing (test 0.888 < paper 97.0).
run.py natively resumes from the latest on-disk checkpoint (checkpoint_18;
save_only_improve:False saved every epoch), so this driver just re-launches the
same save_path/name via the continuation config (num_epochs 33) and it picks up
at internal epoch 18, running stages 4/5/6 to the asymptote.

DIFFERENCES vs run_session_a.py (deliberate):
  - Coconut ONLY (CoT + w/o-thought already landed; their sentinels/checkpoints
    are untouched).
  - PER_RUN_MAX_HOURS = 24  (the 16h cap is what truncated us; ~15 epochs of
    slowing latent stages need more headroom, and supervisor wakes + the WatchRun
    terminate_command remain the real teardown guard).
  - Training log opened in APPEND mode: the resumed run's per-epoch
    "Accuracy on validation set" lines are appended after the original epochs
    0..17, so the k-th val line still maps to checkpoint_{k+1} (continuous log).
    This keeps the best-checkpoint parse identical to the base driver.
Everything else (parse best val -> map to checkpoint_{k+1} -> test-set eval ->
copy best ckpt + META -> sentinels) is identical to run_session_a.py.
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

PER_RUN_MAX_HOURS = 24.0  # generous backstop; supervisor wakes are the real stop
TEST_PATH = "data/prosqa_test.json"

RUN = {
    "name": "prosqa_coconut",
    "cfg": "args_a10/prosqa_coconut_a10_fp32_continue.yaml",
    "mode": "latent",
}


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


def run_torchrun(cfg_path, log_path, max_hours, append=False):
    cmd = ["torchrun", "--nnodes", "1", "--nproc_per_node", "1", "run.py", cfg_path]
    with open(log_path, "a" if append else "w") as lf:
        lf.write(f"\n# continue-driver: launching {' '.join(cmd)} at {time.ctime()}\n")
        lf.flush()
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=ENV)
        deadline = time.time() + max_hours * 3600
        while p.poll() is None:
            if time.time() > deadline:
                lf.write(f"\n# continue-driver: PER_RUN_MAX_HOURS ({max_hours}h) hit — killing.\n")
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


def main():
    name, cfg, mode = RUN["name"], RUN["cfg"], RUN["mode"]
    train_log = f"logs/{name}_train.log"
    eval_log = f"logs/{name}_eval.log"

    print(f"=== [{name}] CONTINUE TRAIN START {time.ctime()} ===", flush=True)
    run_torchrun(cfg, train_log, PER_RUN_MAX_HOURS, append=True)
    print(f"=== [{name}] CONTINUE TRAIN ENDED {time.ctime()} ===", flush=True)

    accs = parse_val_accs(train_log)
    meta = {"name": name, "cfg": cfg, "mode": mode, "val_trajectory": accs,
            "continuation": True}
    if not accs:
        meta["status"] = "ERROR_no_val_lines"
        json.dump(meta, open(f"best_ckpts/{name}_META.json", "w"), indent=2)
        open(f"RUN_{name}_DONE.txt", "w").write("ERROR: no validation lines parsed\n")
        open("SESSION_A_DONE.txt", "w").write("session A continue: ERROR no val lines\n")
        return

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
        eval_cfg = f"args_a10/{name}_continue_eval_gen.yaml"
        write_eval_cfg(cfg, eval_cfg, best_ckpt_abs, mode)
        print(f"=== [{name}] TEST EVAL START {time.ctime()} ===", flush=True)
        run_torchrun(eval_cfg, eval_log, 3.0, append=False)
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
        f"{name}(continue): best_val={best_val:.4f}@ep{best_k+1} test_acc={test_acc} ckpt={best_ckpt}\n")
    open("SESSION_A_DONE.txt", "w").write(
        "session A complete (coconut continued to asymptote)\n" + json.dumps(meta, indent=2))
    print(f"=== SESSION A (continue) DONE {time.ctime()} ===", flush=True)


if __name__ == "__main__":
    main()
