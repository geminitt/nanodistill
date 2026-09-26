"""Teacher data on Modal: Qwen3-8B answers each training prompt, with and without a reasoning trace.

    pixi run modal run src/nanodistill/teacher.py --mode direct --limit 200     # pilot
    pixi run modal run --detach src/nanodistill/teacher.py --mode direct         # everything, detached

Modes:
- direct: non-thinking (the empty think block is part of the prompt), greedy, with the top-20 log-probabilities
  of every generated token, for sequence-level and logit-level distillation.
- cot: thinking mode, sampled as Qwen recommends for it (temperature 0.6, top-p 0.95, top-k 20), for the
  reasoning-trace conditions.

Work is split into shards of SHARD prompts; each finished shard is written to the volume at once, and a rerun
skips the shards that exist, so an interrupted job resumes where it stopped.
"""

import json
import os
import time

import modal

TEACHER = "Qwen/Qwen3-8B"
SHARD = 500
GPU = os.environ.get("NANODISTILL_GPU", "A10G")

app = modal.App("nanodistill-teacher")
image = modal.Image.debian_slim(python_version="3.12").pip_install("vllm==0.30.0", "numpy")
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
volume = modal.Volume.from_name("nanodistill", create_if_missing=True)


@app.function(image=image, gpu=GPU, timeout=6 * 3600, volumes={"/root/.cache/huggingface": hf_cache, "/vol": volume},
              secrets=[modal.Secret.from_name("huggingface")])
def generate(mode: str, records_path: str, out_dir: str) -> dict:
    import numpy as np
    from vllm import LLM, SamplingParams

    records = [json.loads(line) for line in open(records_path)]
    os.makedirs(out_dir, exist_ok=True)
    llm = LLM(TEACHER, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.92, seed=0)
    if mode == "direct":
        params = SamplingParams(temperature=0.0, max_tokens=512, logprobs=20)
    else:
        params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=3072, seed=0)

    stats = {"shards_done": 0, "shards_skipped": 0, "seconds": 0.0, "prompt_tokens": 0, "output_tokens": 0}
    for start in range(0, len(records), SHARD):
        path = f"{out_dir}/shard_{start // SHARD:04d}.jsonl"
        if os.path.exists(path):
            stats["shards_skipped"] += 1
            continue
        batch = records[start:start + SHARD]
        t0 = time.time()
        outputs = llm.generate([r["prompt"] for r in batch], params)
        stats["seconds"] += time.time() - t0
        rows, top_ids, top_lp = [], [], []
        for r, o in zip(batch, outputs):
            gen = o.outputs[0]
            stats["prompt_tokens"] += len(o.prompt_token_ids)
            stats["output_tokens"] += len(gen.token_ids)
            rows.append({"id": r["id"], "kind": r["kind"], "text": gen.text, "token_ids": list(gen.token_ids),
                         "finish_reason": gen.finish_reason})
            if mode == "direct":   # top-20 (id, log-probability) at every generated position
                ids = np.zeros((len(gen.token_ids), 20), np.int32)
                lps = np.full((len(gen.token_ids), 20), -np.inf, np.float32)
                for t, cand in enumerate(gen.logprobs):
                    best = sorted(cand.items(), key=lambda kv: kv[1].rank)[:20]
                    ids[t, :len(best)] = [k for k, _ in best]
                    lps[t, :len(best)] = [v.logprob for _, v in best]
                top_ids.append(ids)
                top_lp.append(lps.astype(np.float16))
        if mode == "direct":
            np.savez(path.replace(".jsonl", ".npz"), lengths=np.array([len(x) for x in top_ids]),
                     ids=np.concatenate(top_ids), logprobs=np.concatenate(top_lp))
        with open(path + ".tmp", "w") as f:           # the .jsonl marks the shard done, so write it last
            f.writelines(json.dumps(row) + "\n" for row in rows)
        os.replace(path + ".tmp", path)
        volume.commit()
        stats["shards_done"] += 1
        print(f"{mode} shard {start // SHARD}: {len(batch)} prompts in {time.time() - t0:.0f}s", flush=True)
    return stats


@app.local_entrypoint()
def main(mode: str = "direct", limit: int = 0, n_train: int = 5000, n_irrelevant: int = 1000):
    from nanodistill.data import EMPTY_THINK, load_examples, prompt, split

    banned = set(json.load(open("data/bfcl_function_names.json")))
    s = split(load_examples(), n_train, n_irrelevant, banned)
    records = [{"id": e["id"], "kind": "train", "prompt": prompt(e["tools"], e["query"])} for e in s["train"]]
    if mode == "direct":   # irrelevance cases need only the direct answer; the empty think block ends every prompt
        records += [{"id": e["id"], "kind": "irrelevant", "prompt": prompt(e["tools"], e["query"])}
                    for e in s["irrelevant"]]
        for r in records:
            r["prompt"] += EMPTY_THINK
    if limit:
        records = records[:limit]
    tag = f"{mode}_n{len(records)}"                  # full runs go to /vol/teacher/{mode}, pilots to /vol/pilot
    local = f"/tmp/records_{tag}.jsonl"
    with open(local, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)
    with volume.batch_upload(force=True) as up:
        up.put_file(local, f"/inputs/{tag}.jsonl")
    out_dir = f"/vol/pilot/{tag}" if limit else f"/vol/teacher/{mode}"
    stats = generate.remote(mode, f"/vol/inputs/{tag}.jsonl", out_dir)
    print(json.dumps(stats, indent=2))
