"""Train the Qwen3-0.6B-Base student on Modal, one (condition, seed) per container.

    pixi run modal run src/nanodistill/train.py --conditions seq_kd_filtered --seeds 0 --steps 20   # smoke test
    pixi run modal run --detach src/nanodistill/train.py --conditions all --seeds 0,1,2

Every condition trains for the same number of optimizer steps on the same number of sequences per step, from
the same starting weights; only what the student learns from differs (see CONDITIONS). The seed sets the data
order. A run saves its state every SAVE_EVERY steps and resumes from it if the container is restarted.
"""

import json
import math
import os
import random

import modal

STUDENT = "Qwen/Qwen3-0.6B-Base"
TOKENIZER = "Qwen/Qwen3-0.6B"        # same vocabulary; its chat template is the one BFCL assumes
STEPS, SEQS_PER_STEP, MICRO = 400, 32, 4
LR, WARMUP, MAX_LEN, SAVE_EVERY = 2e-5, 20, 4096, 100
GPU = os.environ.get("NANODISTILL_TRAIN_GPU", "A100-40GB")

CONDITIONS = {
    # name: what the student learns from
    "sft_gold": "xLAM's verified calls, no teacher",
    "seq_kd": "the teacher's direct answers, all of them",
    "seq_kd_filtered": "the teacher's direct answers that match the verified calls",
    "logit_kd": "the teacher's top-20 distribution over the same sequences as seq_kd",
    "cot_full": "the teacher's reasoning and calls (filtered); the student reasons at test time too",
    "cot_train_only": "seq_kd_filtered plus the filtered reasoning traces; the student does not reason at test time",
    "irrelevance": "seq_kd_filtered plus the teacher's refusals on queries whose tools were removed",
}

app = modal.App("nanodistill-train")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.0", "transformers==5.17.0", "numpy", "datasets", "safetensors")
         .add_local_python_source("nanodistill"))
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
volume = modal.Volume.from_name("nanodistill", create_if_missing=True)


def build(condition: str, teacher_dir: str, tok) -> list[dict]:
    """Training sequences for one condition: prompt ids, completion ids, and (logit_kd) teacher top-20."""
    import numpy as np

    from nanodistill.data import EMPTY_THINK, completion, load_examples, parse_calls, prompt, same_calls, split

    banned = set(json.load(open("/vol/inputs/bfcl_function_names.json")))
    s = split(load_examples(), 5000, 1000, banned)
    gold = {e["id"]: e for e in s["train"]}
    irr = {e["id"]: e for e in s["irrelevant"]}
    enc = lambda text: tok(text, add_special_tokens=False)["input_ids"]

    def teacher(mode):
        rows, tops = [], []
        d = f"{teacher_dir}/{mode}"
        for name in sorted(f for f in os.listdir(d) if f.endswith(".jsonl")):
            rows += [json.loads(line) for line in open(f"{d}/{name}")]
            if mode == "direct":
                z = np.load(f"{d}/{name.replace('.jsonl', '.npz')}")
                offs = np.concatenate([[0], np.cumsum(z["lengths"])])
                tops += [(z["ids"][a:b], z["logprobs"][a:b].astype(np.float32)) for a, b in zip(offs[:-1], offs[1:])]
        return rows, tops

    out = []
    direct_prompt = lambda e: enc(prompt(e["tools"], e["query"]) + EMPTY_THINK)
    if condition == "sft_gold":
        for e in s["train"]:
            out.append({"prompt": direct_prompt(e), "target": enc(completion(e["calls"])[len(EMPTY_THINK):])})
        return out
    rows, tops = teacher("direct")
    direct = [(r, t) for r, t in zip(rows, tops) if r["kind"] == "train"]
    ok = lambda r, calls: r["finish_reason"] == "stop" and same_calls(parse_calls(r["text"]), calls)
    if condition in ("seq_kd", "logit_kd"):
        for r, t in direct:
            item = {"prompt": direct_prompt(gold[r["id"]]), "target": r["token_ids"]}
            if condition == "logit_kd":
                item["top_ids"], item["top_lp"] = t
            out.append(item)
        return out
    filtered = [{"prompt": direct_prompt(gold[r["id"]]), "target": r["token_ids"]}
                for r, _ in direct if ok(r, gold[r["id"]]["calls"])]
    if condition == "seq_kd_filtered":
        return filtered
    if condition == "irrelevance":
        refusals = [{"prompt": direct_prompt(irr[r["id"]]), "target": r["token_ids"]}
                    for r in rows if r["kind"] == "irrelevant" and r["finish_reason"] == "stop"
                    and not parse_calls(r["text"])]
        return filtered + refusals
    cot_rows, _ = teacher("cot")
    cot = [{"prompt": enc(prompt(gold[r["id"]]["tools"], gold[r["id"]]["query"])), "target": r["token_ids"]}
           for r in cot_rows if ok(r, gold[r["id"]]["calls"])]
    if condition == "cot_full":
        return cot
    if condition == "cot_train_only":
        return filtered + cot
    raise ValueError(condition)


@app.function(image=image, gpu=GPU, timeout=4 * 3600, volumes={"/root/.cache/huggingface": hf_cache, "/vol": volume},
              secrets=[modal.Secret.from_name("huggingface")])
def train(condition: str, seed: int, steps: int = STEPS, teacher_dir: str = "/vol/teacher") -> dict:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = f"/vol/students/{condition}_s{seed}" + ("" if steps == STEPS else f"_steps{steps}")
    if os.path.exists(f"{out_dir}/model.safetensors"):
        return {"condition": condition, "seed": seed, "status": "done already"}
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    data = build(condition, teacher_dir, tok)
    data = [d for d in data if len(d["prompt"]) + len(d["target"]) <= MAX_LEN]
    model = AutoModelForCausalLM.from_pretrained(STUDENT, torch_dtype=torch.float32).cuda()
    model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / WARMUP) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    ckpt, start, log = f"{out_dir}/resume.pt", 0, []
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location="cuda")
        model.load_state_dict(state["model"]); opt.load_state_dict(state["opt"]); sched.load_state_dict(state["sched"])
        start, log = state["step"], state["log"]
    rng = random.Random(seed)
    order = []
    while len(order) < steps * SEQS_PER_STEP:            # same data order for a seed, epoch after epoch
        epoch = list(range(len(data))); rng.shuffle(epoch); order += epoch
    model.train()
    for step in range(start, steps):
        batch = [data[i] for i in order[step * SEQS_PER_STEP:(step + 1) * SEQS_PER_STEP]]
        total_tokens = sum(len(d["target"]) for d in batch)
        loss_sum = 0.0
        for m in range(0, SEQS_PER_STEP, MICRO):
            micro = batch[m:m + MICRO]
            T = max(len(d["prompt"]) + len(d["target"]) for d in micro)
            x = torch.full((len(micro), T), tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros((len(micro), T), dtype=torch.bool)
            attn = torch.zeros((len(micro), T), dtype=torch.long)
            for r, d in enumerate(micro):
                seq = d["prompt"] + d["target"]
                x[r, :len(seq)] = torch.tensor(seq)
                attn[r, :len(seq)] = 1
                mask[r, len(d["prompt"]):len(seq)] = True   # positions whose token is a target
            x, mask, attn = x.cuda(), mask.cuda(), attn.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x, attention_mask=attn).logits[:, :-1].float()
            logp = F.log_softmax(logits, -1)
            tgt_mask = mask[:, 1:]
            if "top_ids" in micro[0]:
                loss = 0.0
                for r, d in enumerate(micro):             # soft targets: -sum p_T log p_S over the teacher's top 20
                    p0 = len(d["prompt"]) - 1
                    ids = torch.tensor(d["top_ids"], device="cuda").long()
                    p_t = torch.softmax(torch.tensor(d["top_lp"], device="cuda"), -1)
                    lp_s = logp[r, p0:p0 + len(d["target"])].gather(-1, ids)
                    loss = loss - (p_t * lp_s).sum()
            else:
                y = x[:, 1:]
                loss = -(logp.gather(-1, y.unsqueeze(-1)).squeeze(-1) * tgt_mask).sum()
            (loss / total_tokens).backward()
            loss_sum += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        log.append({"step": step + 1, "loss": loss_sum / total_tokens, "lr": sched.get_last_lr()[0]})
        if (step + 1) % 10 == 0:
            print(f"{condition} s{seed} step {step + 1}/{steps} loss {log[-1]['loss']:.4f}", flush=True)
        if (step + 1) % SAVE_EVERY == 0 and step + 1 < steps:
            os.makedirs(out_dir, exist_ok=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step + 1, "log": log}, ckpt)
            volume.commit()
    model = model.to(torch.bfloat16)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    json.dump({"condition": condition, "seed": seed, "steps": steps, "examples": len(data), "log": log},
              open(f"{out_dir}/train_log.json", "w"))
    if os.path.exists(ckpt):
        os.remove(ckpt)
    volume.commit()
    return {"condition": condition, "seed": seed, "examples": len(data), "final_loss": log[-1]["loss"]}


@app.local_entrypoint()
def main(conditions: str = "all", seeds: str = "0,1,2", steps: int = STEPS):
    names = list(CONDITIONS) if conditions == "all" else conditions.split(",")
    with volume.batch_upload(force=True) as up:
        up.put_file("data/bfcl_function_names.json", "/inputs/bfcl_function_names.json")
    jobs = [(c, int(s)) for c in names for s in seeds.split(",")]
    for result in train.starmap([(c, s, steps) for c, s in jobs]):
        print(json.dumps(result))
