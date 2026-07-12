#!/usr/bin/env python3
"""GSM8K session driver — the three Table-1 cells (GPT-2 124M), single seed.

Order (brief): CoT first (its best-val checkpoint INITIALIZES Coconut), then
No-CoT, then Coconut fp32. Runs on ONE Lambda A10, launched DETACHED via
`lambda run_bg` (tmux) so it outlives the SSH session.

For each run it:
  1. trains (torchrun) to num_epochs, teeing to logs/<name>_train.log;
  2. parses every "Accuracy on validation set" line, picks the best, and maps it
     to a checkpoint accounting for the config's `resume` offset:
        checkpoint number = resume + best_k + 1
     (run.py saves checkpoint_{epoch+1}; the loop starts at epoch=resume, so the
      k-th 0-indexed val line is epoch=resume+k -> checkpoint_{resume+k+1});
  3. writes an eval config (only_eval:True, load_model_path=best ckpt,
     val_path=gsm_test.json) and evals on the TEST set (paper-comparable);
  4. copies the ONE best checkpoint to best_ckpts/<name>/ + writes META.json;
  5. touches RUN_<name>_DONE.txt.

CoT->Coconut init: after the CoT run, the best-val CoT checkpoint's ABSOLUTE path
is patched into args_a10/gsm_coconut_a10_fp32.yaml's load_model_path line before
the Coconut run launches (README: "~40% val checkpoint as the Coconut init").

DESIGN NOTES:
- torchrun exit code IGNORED on purpose; a run can finish all epochs OR be
  early-cut via pkill once val is flat near target. Either way the per-epoch
  checkpoints are on disk (CoT/No-CoT: save_only_improve keeps the best;
  Coconut: save_only_improve:False saves every epoch) so "parse the log + eval
  the best" is correct.
- Eval `resume` per run: CoT/No-CoT -> 0 (no latent); Coconut -> 15 (fully-latent
  stage 3, since 15//epochs_per_stage=5 clamps to max_latent_stage=3, and
  range(15,25) is non-empty so the only_eval loop actually runs — resume:40 would
  make range(40,25) EMPTY and skip eval on GSM's 25-epoch configs).
- Coconut is run in ONE continuous shot (never restarted): reset_optimizer:True at
  a stage boundary on a cold restart is exactly what NaN'd the ProsQA resume.
- Completion is signalled by SENTINEL FILES only, never live progress
  (2026-06-19 idle-bill lesson).
- The box CANNOT self-terminate (Lambda API key not on the box); teardown is
  enforced from the homestation via WatchRun + supervisor wakes on the sentinel.
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

TEST_PATH = "data/gsm_test.json"

# Generous per-run wall-clock backstops (GSM epochs ~26 min; the prior ProsQA run
# was truncated by a too-short 16h cap — do NOT repeat that). These are runaway
# guards, not done-signals; the real teardown is the homestation on the sentinel.
RUNS = [
    {
        "name": "gsm_cot",
        "cfg": "args_a10/gsm_cot_a10.yaml",          # bf16 baseline; also the Coconut init
        "mode": "cot",
        "eval_resume": 0,
        "max_hours": 20.0,
    },
    {
        "name": "gsm_nocot",
        "cfg": "args_a10/gsm_nocot_a10.yaml",         # bf16 baseline
        "mode": "cot",
        "eval_resume": 0,
        "max_hours": 16.0,
    },
    {
        "name": "gsm_coconut",
        "cfg": "args_a10/gsm_coconut_a10_fp32.yaml",  # fp32 latent; load_model_path patched below
        "mode": "latent",
        "eval_resume": 15,
        "max_hours": 34.0,
    },
]

COCONUT_CFG = "args_a10/gsm_coconut_a10_fp32.yaml"


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


def cfg_resume(cfg_path):
    d, _ = read_yaml_flat(cfg_path)
    return int(d.get("resume", "0"))


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
        lf.write(f"\n# gsm-driver: launching {' '.join(cmd)} at {time.ctime()}\n")
        lf.flush()
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=ENV)
        deadline = time.time() + max_hours * 3600
        while p.poll() is None:
            if time.time() > deadline:
                lf.write(f"\n# gsm-driver: PER_RUN_MAX_HOURS ({max_hours}h) hit — killing.\n")
                lf.flush()
                p.terminate()
                try:
                    p.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    p.kill()
                break
            time.sleep(20)


def write_eval_cfg(train_cfg, out_cfg, best_ckpt_abs, eval_resume):
    d, order = read_yaml_flat(train_cfg)
    over = {
        "only_eval": "True",
        "load_model_path": best_ckpt_abs,
        "val_path": TEST_PATH,
        "name": d["name"] + "-eval",
        "resume": str(eval_resume),
    }
    d.update(over)
    with open(out_cfg, "w") as f:
        f.write(f"# auto-generated eval config for {d['name']} (GSM test-set eval)\n")
        for k in order:
            f.write(f"{k}: {d[k]}\n")
        for k in over:
            if k not in order:
                f.write(f"{k}: {d[k]}\n")


def patch_coconut_init(cot_best_ckpt_abs):
    """Rewrite the Coconut config's load_model_path to the best-val CoT checkpoint."""
    lines = open(COCONUT_CFG).read().splitlines()
    out = []
    for ln in lines:
        if ln.strip().startswith("load_model_path:"):
            out.append(f"load_model_path: {cot_best_ckpt_abs}")
        else:
            out.append(ln)
    open(COCONUT_CFG, "w").write("\n".join(out) + "\n")
    print(f"=== patched {COCONUT_CFG} load_model_path -> {cot_best_ckpt_abs} ===", flush=True)


def do_run(run):
    name, cfg, mode = run["name"], run["cfg"], run["mode"]
    train_log = f"logs/{name}_train.log"
    eval_log = f"logs/{name}_eval.log"
    print(f"=== [{name}] TRAIN START {time.ctime()} ===", flush=True)
    run_torchrun(cfg, train_log, run["max_hours"], append=False)
    print(f"=== [{name}] TRAIN ENDED {time.ctime()} ===", flush=True)

    accs = parse_val_accs(train_log)
    resume = cfg_resume(cfg)
    meta = {"name": name, "cfg": cfg, "mode": mode, "resume": resume,
            "val_trajectory": accs}
    if not accs:
        meta["status"] = "ERROR_no_val_lines"
        json.dump(meta, open(f"best_ckpts/{name}_META.json", "w"), indent=2)
        open(f"RUN_{name}_DONE.txt", "w").write("ERROR: no validation lines parsed\n")
        print(f"=== [{name}] ERROR no val lines ===", flush=True)
        return meta

    best_k = max(range(len(accs)), key=lambda i: accs[i])
    best_val = accs[best_k]
    ckpt_num = resume + best_k + 1          # resume-aware mapping
    best_ckpt = f"checkpoint_{ckpt_num}"
    sdir = save_dir_for(cfg)
    best_ckpt_abs = os.path.abspath(os.path.join(sdir, best_ckpt))
    meta.update({"best_epoch_0idx_in_log": best_k, "best_epoch_1idx": ckpt_num,
                 "best_ckpt": best_ckpt, "best_val_acc": best_val,
                 "n_epochs_seen": len(accs), "checkpoint_dir": sdir,
                 "best_ckpt_abs": best_ckpt_abs})
    print(f"=== [{name}] best val {best_val:.4f} (log-idx {best_k}) -> {best_ckpt} ===", flush=True)

    test_acc = None
    if os.path.exists(best_ckpt_abs):
        eval_cfg = f"args_a10/{name}_eval_gen.yaml"
        write_eval_cfg(cfg, eval_cfg, best_ckpt_abs, run["eval_resume"])
        print(f"=== [{name}] TEST EVAL START {time.ctime()} ===", flush=True)
        run_torchrun(eval_cfg, eval_log, 4.0, append=False)
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
        f"{name}: best_val={best_val:.4f} test_acc={test_acc} ckpt={best_ckpt}\n")
    return meta


def main():
    summary = {"session": "GSM", "started": time.ctime(), "runs": []}
    for run in RUNS:
        # Patch the Coconut init from the CoT best checkpoint before its run starts.
        if run["name"] == "gsm_coconut":
            cot_meta_path = "best_ckpts/gsm_cot_META.json"
            if os.path.exists(cot_meta_path):
                cot_meta = json.load(open(cot_meta_path))
                cot_ckpt = cot_meta.get("best_ckpt_abs")
                if cot_ckpt and os.path.exists(cot_ckpt):
                    patch_coconut_init(cot_ckpt)
                else:
                    msg = f"CoT checkpoint missing for Coconut init: {cot_ckpt}"
                    print("=== ERROR:", msg, flush=True)
                    summary["runs"].append({"name": run["name"], "status": f"SKIP: {msg}"})
                    open(f"RUN_{run['name']}_DONE.txt", "w").write(f"SKIP: {msg}\n")
                    json.dump(summary, open("SESSION_GSM_SUMMARY.json", "w"), indent=2)
                    continue
            else:
                msg = "gsm_cot_META.json missing; cannot init Coconut"
                print("=== ERROR:", msg, flush=True)
                summary["runs"].append({"name": run["name"], "status": f"SKIP: {msg}"})
                open(f"RUN_{run['name']}_DONE.txt", "w").write(f"SKIP: {msg}\n")
                json.dump(summary, open("SESSION_GSM_SUMMARY.json", "w"), indent=2)
                continue
        try:
            summary["runs"].append(do_run(run))
        except Exception as e:  # noqa
            summary["runs"].append({"name": run["name"], "status": f"EXCEPTION: {e}"})
            open(f"RUN_{run['name']}_DONE.txt", "w").write(f"EXCEPTION: {e}\n")
        json.dump(summary, open("SESSION_GSM_SUMMARY.json", "w"), indent=2)
    summary["finished"] = time.ctime()
    json.dump(summary, open("SESSION_GSM_SUMMARY.json", "w"), indent=2)
    open("SESSION_GSM_DONE.txt", "w").write("GSM session complete\n" + json.dumps(summary, indent=2))
    print(f"=== SESSION GSM DONE {time.ctime()} ===", flush=True)


if __name__ == "__main__":
    main()
