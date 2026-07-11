#!/usr/bin/env python3
"""From-scratch driver — ProsQA Coconut fp32, trained continuously to the ep30+
fully-latent asymptote.

WHY THIS EXISTS (2026-07-11):
The original Session-A coconut run was cost-truncated at epoch 18 by a 16h
per-run backstop while validation was still climbing (test 0.888 < paper 0.970).
The saved checkpoint_17 (val 0.897) contains model weights + PARTIAL state but NOT
the full Adam moment buffers, so a FAITHFUL warm resume is impossible; every cold
resume (reset_optimizer:True) NaN'd at batch ~32 of the epoch-17->18 stage
boundary (reproduced 3x). The only faithful route to the true asymptote is a
SINGLE continuous run with warm in-process optimizer momentum from ep0 -> ep30+.
That is what this driver does: it re-runs what the original box did, but with a
24h backstop (not 16h) so the stage-6 slowdown near the asymptote can't truncate
it again.

Runs on the Lambda A10, launched DETACHED via `lambda run_bg` (tmux) so it
outlives the SSH session. It:
  1. trains (torchrun) the from-scratch fp32 config to num_epochs (50), teeing to
     logs/prosqa_coconut_train.log;
  2. parses the best VALIDATION accuracy line -> checkpoint_{k+1}
     (resume=0, save_only_improve:False => k-th 0-indexed val line maps to
     checkpoint_{k+1}, verified against run.py);
  3. writes an eval config (only_eval:True, load_model_path=best ckpt,
     val_path=test set, resume high enough to sit in the final fully-latent stage)
     and evals that checkpoint on the TEST set (paper-comparable);
  4. copies the ONE best checkpoint to best_ckpts/prosqa_coconut/ + META.json;
  5. writes RUN_prosqa_coconut_DONE.txt and SESSION_DONE.txt sentinels.

DESIGN NOTES:
- torchrun's exit code is IGNORED on purpose. A run can end by finishing all
  epochs OR by a documented early-cut `pkill` once val is flat near target; both
  leave per-epoch checkpoints on disk (save_only_improve:False saves every epoch),
  so "parse whatever's in the log + eval the best" is correct either way.
- PER_RUN_MAX_HOURS = 24: the 16h cap is what truncated the original; the slowing
  latent stages need headroom. Supervisor wakes + WatchRun remain the real
  teardown guard (the box canNOT self-terminate via the Lambda API).
- Completion is signalled by SENTINEL FILES only (never live progress), per the
  2026-06-19 idle-bill lesson.
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
    "cfg": "args_a10/prosqa_coconut_a10_fp32.yaml",  # from scratch: resume:0, num_epochs:50, bf16:False
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
        lf.write(f"\n# scratch-driver: launching {' '.join(cmd)} at {time.ctime()}\n")
        lf.flush()
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=ENV)
        deadline = time.time() + max_hours * 3600
        while p.poll() is None:
            if time.time() > deadline:
                lf.write(f"\n# scratch-driver: PER_RUN_MAX_HOURS ({max_hours}h) hit — killing.\n")
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

    print(f"=== [{name}] SCRATCH TRAIN START {time.ctime()} ===", flush=True)
    run_torchrun(cfg, train_log, PER_RUN_MAX_HOURS, append=False)
    print(f"=== [{name}] SCRATCH TRAIN ENDED {time.ctime()} ===", flush=True)

    accs = parse_val_accs(train_log)
    meta = {"name": name, "cfg": cfg, "mode": mode, "val_trajectory": accs,
            "from_scratch": True}
    if not accs:
        meta["status"] = "ERROR_no_val_lines"
        json.dump(meta, open(f"best_ckpts/{name}_META.json", "w"), indent=2)
        open(f"RUN_{name}_DONE.txt", "w").write("ERROR: no validation lines parsed\n")
        open("SESSION_DONE.txt", "w").write("coconut scratch: ERROR no val lines\n")
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
        eval_cfg = f"args_a10/{name}_scratch_eval_gen.yaml"
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
        f"{name}(scratch): best_val={best_val:.4f}@ep{best_k+1} test_acc={test_acc} ckpt={best_ckpt}\n")
    open("SESSION_DONE.txt", "w").write(
        "coconut scratch run complete (trained to asymptote)\n" + json.dumps(meta, indent=2))
    print(f"=== COCONUT SCRATCH DONE {time.ctime()} ===", flush=True)


if __name__ == "__main__":
    main()
