"""BFCL v4 single-turn evaluation on Modal, for the baselines and every trained student.

    pixi run modal run src/nanodistill/bfcl_eval_modal.py --models baselines
    pixi run modal run --detach src/nanodistill/bfcl_eval_modal.py --models students

BFCL's own Qwen3 FC handler builds the prompt; the students' prompt and output format was tested against it
(tests/test_bfcl_format.py). Two ways of running a model:
- "think": BFCL's handler as is; Qwen3 thinks before calling (BFCL's standard setting for Qwen3).
- "direct": the same prompt plus the empty think block, i.e. Qwen3's non-thinking mode, which is how the
  students without reasoning traces were trained.
Each (model, mode) gets its own BFCL project folder on the volume; a finished one is not recomputed.
"""

import json
import os
import subprocess
from pathlib import Path

import modal

CATEGORY = "single_turn"          # non-live and live single-turn categories; multi-turn and agentic are left out
GPU = os.environ.get("NANODISTILL_EVAL_GPU", "A10G")

app = modal.App("nanodistill-bfcl")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("bfcl-eval[oss-eval-vllm]==2026.3.23", "numpy==1.26.4", "soundfile", "transformers==4.51.3")   # qwen-agent imports soundfile
         .add_local_file(str(Path(__file__).parent / "bfcl_runner.py"), "/root/bfcl_runner.py")
         .add_local_python_source("nanodistill"))
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
volume = modal.Volume.from_name("nanodistill", create_if_missing=True)


@app.function(image=image, gpu=GPU, timeout=3 * 3600, volumes={"/root/.cache/huggingface": hf_cache, "/vol": volume},
              secrets=[modal.Secret.from_name("huggingface-secret")])
def evaluate(label: str, hf_name: str, mode: str, local_path: str = "") -> dict:
    root = f"/vol/bfcl/{label}_{mode}"
    summary = f"{root}/summary.json"
    if os.path.exists(summary):
        return json.load(open(summary))
    os.makedirs(root, exist_ok=True)
    env = {**os.environ, "BFCL_PROJECT_ROOT": root}
    if local_path:
        from huggingface_hub import snapshot_download

        from nanodistill.bfcl_local import REFERENCE_FILES, compatible_copy
        reference_dir = snapshot_download("Qwen/Qwen3-0.6B", allow_patterns=list(REFERENCE_FILES))
        local_path = compatible_copy(local_path, reference_dir, out="/tmp/student")
    registry = f"nanodistill/{label.replace('_', '-')}-{mode}-FC"   # BFCL maps "_" in folder names back to "/"
    gen = ["generate", "--model", registry, "--test-category", CATEGORY, "--backend", "vllm", "--num-gpus", "1",
           "--gpu-memory-utilization", "0.9"] + (["--local-model-path", local_path] if local_path else [])
    subprocess.run(["python", "/root/bfcl_runner.py", registry, hf_name, mode] + gen, env=env, check=True)
    subprocess.run(["python", "/root/bfcl_runner.py", registry, hf_name, mode,
                    "evaluate", "--model", registry, "--test-category", CATEGORY], env=env, check=True)
    scores = {}
    for dirpath, _, files in os.walk(f"{root}/score"):
        for f in files:
            if f.endswith("_score.json"):
                head = json.loads(open(os.path.join(dirpath, f)).readline())
                scores[f.replace("BFCL_v4_", "").replace("_score.json", "")] = head
    out = {"label": label, "mode": mode, "hf_name": hf_name, "scores": scores}
    json.dump(out, open(summary, "w"), indent=1)
    volume.commit()
    return out


BASELINES = [("qwen3-0.6b", "Qwen/Qwen3-0.6B"), ("qwen3-8b", "Qwen/Qwen3-8B")]


@app.local_entrypoint()
def main(models: str = "baselines", modes: str = "direct,think"):
    jobs = []
    if models == "baselines":
        jobs = [(label, hf, mode, "") for label, hf in BASELINES for mode in modes.split(",")]
    else:   # every finished student, in the mode it was trained for
        for entry in volume.listdir("/students"):
            name = entry.path.split("/")[-1]
            mode = "think" if name.startswith("cot_full") else "direct"
            jobs.append((name, "Qwen/Qwen3-0.6B", mode, f"/vol/students/{name}"))
    # BFCL serves each model at its full 40,960-token context: Qwen3-8B's KV cache for that does not fit next to
    # its 16 GB of weights on a 24 GB A10G, so the 8B baseline runs on a 48 GB L40S.
    calls = [(evaluate.with_options(gpu="L40S") if "8b" in j[0] else evaluate).spawn(*j) for j in jobs]
    for result in (c.get() for c in calls):
        print(json.dumps({"label": result["label"], "mode": result["mode"],
                          "scores": {k: v.get("accuracy") for k, v in result["scores"].items()}}))
