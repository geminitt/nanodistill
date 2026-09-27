"""BFCL v4 single-turn evaluation on the local GPU (a 6 GB card is enough for a 0.6B model).

    pixi run -e eval python -m nanodistill.bfcl_local --label qwen3-0.6b --model Qwen/Qwen3-0.6B --mode direct
    pixi run -e eval python -m nanodistill.bfcl_local --label cot_full_s0 --model runs/students/cot_full_s0 --mode think

Results go to runs/bfcl/{label}_{mode}; a finished one (summary.json) is not recomputed, and an interrupted one
resumes, because BFCL only generates the items missing from its result files.

Context. BFCL serves the model at the context length in its config (40,960 tokens for Qwen3), and vLLM
reserves KV cache for one full sequence: 28 layers x 8 KV heads x 128 dims x 2 (K, V) x 2 bytes = 115 KB per
token, 4.7 GB in all, which does not fit next to the weights on 6 GB. The evaluation copy caps it at MAX_LEN
(1.3 GB of KV cache). This changes nothing: BFCL asks for min(4096, context - prompt) new tokens, and the
longest single-turn prompt is 6,178 tokens, so every item keeps its 4,096-token budget.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
REFERENCE_FILES = TOKENIZER_FILES + ("generation_config.json",)   # taken from Qwen3-0.6B for every 0.6B model
MAX_LEN = 12288   # >= longest prompt (6,178) + BFCL's 4,096-token budget; the KV cache for it fits 6 GB
CATEGORY = "single_turn"          # non-live and live single-turn categories; multi-turn and agentic are left out


def compatible_copy(model_dir: str, reference_dir: str, out: str, max_len: int | None = None) -> str:
    """A copy of the model that the evaluation stack reads exactly as training did; model_dir is never written.

    - Weights are copied, never linked: an earlier version linked them and then copied every file of a
      Hugging Face snapshot, whose model.safetensors went through the link and replaced the students'
      weights with Qwen3-0.6B's.
    - Config: students are saved by transformers 5, which keeps the RoPE base in `rope_parameters`.
      transformers 4.51 (vLLM 0.8.5) reads only the top-level `rope_theta` and silently falls back to
      10,000 instead of Qwen3's 1,000,000, which garbles the output; both keys are written.
    - Tokenizer and generation settings come from Qwen3-0.6B (reference_dir). The tokenizer was never
      changed, and transformers 4.51 cannot parse the one transformers 5 saves. Base's generation config
      would stop only at <|endoftext|> and cap output at 2,048 tokens, while Qwen3-0.6B stops at <|im_end|>
      too and leaves BFCL's 4,096-token budget in force; every 0.6B model gets the same settings.
    """
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    for f in os.listdir(model_dir):
        if f.endswith(".safetensors"):
            shutil.copyfile(os.path.join(model_dir, f), os.path.join(out, f))
    config = json.load(open(os.path.join(model_dir, "config.json")))
    config["torch_dtype"] = "bfloat16"
    rope = config.get("rope_parameters")
    if rope:
        config["rope_theta"] = rope["rope_theta"]
        config["rope_scaling"] = None if rope.get("rope_type", "default") == "default" else rope
    if max_len:
        config["max_position_embeddings"] = max_len
    json.dump(config, open(os.path.join(out, "config.json"), "w"), indent=1)
    for f in REFERENCE_FILES:
        shutil.copyfile(os.path.join(reference_dir, f), os.path.join(out, f))
    return out


def summarize(root: str) -> dict:
    scores = {}
    for dirpath, _, files in os.walk(f"{root}/score"):
        for f in files:
            if f.endswith("_score.json"):
                scores[f.replace("BFCL_v4_", "").replace("_score.json", "")] = json.loads(
                    open(os.path.join(dirpath, f)).readline())
    return scores


def evaluate(label: str, model: str, mode: str, runs: str = "runs") -> dict:
    root = os.path.abspath(f"{runs}/bfcl/{label}_{mode}")
    summary = f"{root}/summary.json"
    if os.path.exists(summary):
        return json.load(open(summary))
    from huggingface_hub import snapshot_download

    reference_dir = snapshot_download("Qwen/Qwen3-0.6B", allow_patterns=list(REFERENCE_FILES))
    model_dir = model if os.path.isdir(model) else snapshot_download(model)
    local = compatible_copy(model_dir, reference_dir, os.path.abspath(f"{runs}/eval_copy/{label}"), MAX_LEN)
    os.makedirs(root, exist_ok=True)
    env = {**os.environ, "BFCL_PROJECT_ROOT": root}
    registry = f"nanodistill/{label.replace('_', '-')}-{mode}-FC"   # BFCL maps "_" in folder names back to "/"
    runner = [sys.executable, os.path.join(os.path.dirname(__file__), "bfcl_runner.py"), registry,
              "Qwen/Qwen3-0.6B", mode]
    subprocess.run(runner + ["generate", "--model", registry, "--test-category", CATEGORY, "--backend", "vllm",
                             "--num-gpus", "1", "--gpu-memory-utilization", "0.92", "--local-model-path", local],
                   env=env, check=True)
    subprocess.run(runner + ["evaluate", "--model", registry, "--test-category", CATEGORY], env=env, check=True)
    out = {"label": label, "mode": mode, "model": model, "max_len": MAX_LEN, "scores": summarize(root)}
    json.dump(out, open(summary, "w"), indent=1)
    shutil.rmtree(local, ignore_errors=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", required=True, help="a Hugging Face model id or a local model folder")
    ap.add_argument("--mode", choices=["direct", "think"], required=True)
    args = ap.parse_args()
    result = evaluate(args.label, args.model, args.mode)
    print(json.dumps({k: v.get("accuracy") for k, v in result["scores"].items()}))


if __name__ == "__main__":
    main()
