"""Turn BFCL outputs into the pre-registered comparisons (README, "Pre-registered analysis").

    pixi run python -m nanodistill.analysis --bfcl runs/bfcl --out results

Per-item correctness comes from BFCL's own files: an item is correct unless its id is listed in the category's
score file (which records every failure). Output tokens come from the result files.
"""

import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

import numpy as np

PRIMARY = ["simple_python", "multiple", "parallel", "parallel_multiple"]
LIVE = ["live_simple", "live_multiple", "live_parallel", "live_parallel_multiple"]
IRRELEVANCE = ["irrelevance", "live_irrelevance"]
GROUPS = {"primary": PRIMARY, "live AST": LIVE, "irrelevance": IRRELEVANCE}   # primary: non-live Python AST
# The seven pre-registered tests (README, "Comparisons"), each on its own metric. One Holm correction runs across
# those whose two conditions were evaluated (5 of the 7, the study was closed without the others); a test is
# resolved when both its item-bootstrap and its per-seed p-values pass after correction.
# (question, A, B, metric group, null difference A - B)
MARGIN = 0.02     # non-inferiority margin for `irrelevance` on the primary metric
FAMILY = [
    ("teacher vs dataset answers", "seq_kd_filtered", "sft_gold", "primary", 0.0),
    ("filtering the teacher's answers", "seq_kd_filtered", "seq_kd", "primary", 0.0),
    ("soft vs hard targets", "logit_kd", "seq_kd", "primary", 0.0),
    ("reasoning traces, also at test time", "cot_full", "seq_kd_filtered", "primary", 0.0),
    ("reasoning traces, training only", "cot_train_only", "seq_kd_filtered", "primary", 0.0),
    ("refusal examples teach not calling", "irrelevance", "seq_kd_filtered", "irrelevance", 0.0),
    ("refusal examples cost at most 2 points (non-inferiority)", "irrelevance", "seq_kd_filtered", "primary", -MARGIN),
]


def load_run(folder: str) -> dict:
    """{category: {"correct": {id: bool}, "tokens": {id: output tokens}, "unparsed": n}} for one BFCL folder."""
    out = {}
    for path in glob.glob(f"{folder}/result/*/*/BFCL_v4_*_result.json"):
        cat = re.search(r"BFCL_v4_(.+)_result\.json", path)[1]
        rows = [json.loads(line) for line in open(path)]
        score = path.replace("/result/", "/score/").replace("_result.json", "_score.json")
        failed = {json.loads(line)["id"]: json.loads(line) for line in list(open(score))[1:]}
        out[cat] = {"correct": {r["id"]: r["id"] not in failed for r in rows},
                    "tokens": {r["id"]: r.get("output_token_count", 0) for r in rows},
                    "unparsed": sum("decoder_failed" in str(f.get("error_type", "")) for f in failed.values())}
    return out


def accuracy(run: dict, cats: list[str]) -> float:
    vals = [v for c in cats for v in run[c]["correct"].values()]
    return float(np.mean(vals))


def item_vector(runs: list[dict], cats: list[str]) -> tuple[list, np.ndarray]:
    """Per-item correctness averaged over seeds, in a fixed item order."""
    ids = [(c, i) for c in cats for i in sorted(runs[0][c]["correct"])]
    return ids, np.array([[r[c]["correct"][i] for c, i in ids] for r in runs], float).mean(0)


def t_test(diffs: list[float]) -> float:
    """Two-sided one-sample t test p-value (Student's t, n - 1 degrees of freedom)."""
    from math import lgamma
    d = np.asarray(diffs, float)
    n = len(d)
    if n < 2 or d.std(ddof=1) == 0:
        return 0.0 if abs(d.mean()) > 0 else 1.0
    t = abs(d.mean()) / (d.std(ddof=1) / math.sqrt(n))
    df = n - 1

    def betainc(a, b, x, terms=400):                 # regularized incomplete beta by continued fraction
        if x <= 0 or x >= 1:
            return float(x >= 1)
        if x > (a + 1) / (a + b + 2):
            return 1 - betainc(b, a, 1 - x)
        front = math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * math.log(x) + b * math.log1p(-x))
        c, dd = 1.0, 1 / (1 - (a + b) * x / (a + 1))
        f = dd
        for m in range(1, terms):
            for num in (m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
                        -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))):
                dd = 1 / (1 + num * dd); c = 1 + num / c; f *= c * dd
        return front * f / a

    return betainc(df / 2, 0.5, df / (df + t * t))


def holm(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adj, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adj[i] = running
    return adj


def compare(a_runs: list[dict], b_runs: list[dict], cats: list[str], null: float = 0.0, n_boot: int = 10_000,
            seed: int = 0) -> dict:
    """A - B against `null`: paired bootstrap over items (seed-averaged) and a t test on the per-seed differences.

    Both p-values are two-sided tests of "the difference equals `null`"; a resolved non-inferiority test also
    needs the difference above `null`.
    """
    _, a = item_vector(a_runs, cats)
    _, b = item_vector(b_runs, cats)
    diff = a - b
    rng = np.random.default_rng(seed)
    boot = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p_boot = min(1.0, 2 * min((boot <= null).mean(), (boot >= null).mean()))
    per_seed = [accuracy(x, cats) - accuracy(y, cats) for x, y in zip(a_runs, b_runs)]
    return {"diff": float(diff.mean()), "ci95": [float(lo), float(hi)], "p_boot": float(p_boot),
            "per_seed": per_seed, "p_seed": t_test([d - null for d in per_seed])}


def load_all(bfcl_dir: str) -> dict:
    runs = defaultdict(dict)
    for folder in sorted(glob.glob(f"{bfcl_dir}/*")):
        name = os.path.basename(folder)
        m = re.fullmatch(r"(.+)_s(\d+)_(direct|think)", name)
        if m:
            runs[m[1]][int(m[2])] = load_run(folder)
        elif re.fullmatch(r"qwen3-.+_(direct|think)", name):
            runs["baseline:" + name][0] = load_run(folder)
    return runs


def report(runs: dict) -> str:
    lines = ["# Results", "", "## Accuracy by model (mean over seeds; baselines have one run)", "",
             "| model | seeds | " + " | ".join(GROUPS) + " | output tokens (median) | unparseable |",
             "|---|---|" + "---|" * len(GROUPS) + "---|---|"]
    for name, by_seed in sorted(runs.items()):
        rs = [by_seed[s] for s in sorted(by_seed)]
        if not all(all(c in r for c in PRIMARY + LIVE + IRRELEVANCE) for r in rs):
            continue
        accs = [f"{100 * np.mean([accuracy(r, cats) for r in rs]):.1f}" for cats in GROUPS.values()]
        toks = np.median([t for r in rs for c in PRIMARY for t in r[c]["tokens"].values()])
        unparsed = np.mean([sum(r[c]["unparsed"] for c in r) for r in rs])
        lines.append(f"| {name} | {len(rs)} | " + " | ".join(accs) + f" | {toks:.0f} | {unparsed:.0f} |")

    def paired(a, b):
        seeds = sorted(set(runs.get(a, {})) & set(runs.get(b, {})))
        return ([runs[a][s] for s in seeds], [runs[b][s] for s in seeds]) if len(seeds) >= 2 else None

    lines += ["", "## Pre-registered tests", "",
              "Resolved = the item-bootstrap p-value **and** the per-seed t-test p-value are both below 0.05 after "
              "one Holm correction across the tests in the table (and, for non-inferiority, the difference is above "
              f"-{100 * MARGIN:.0f} points).", ""]
    done = [(t, compare(*paired(t[1], t[2]), GROUPS[t[3]], null=t[4])) for t in FAMILY if paired(t[1], t[2])]
    if len(done) < len(FAMILY):
        lines.append(f"Only {len(done)} of the {len(FAMILY)} tests have both conditions evaluated (the study was "
                     "closed without the others); the correction below covers those only.")
        lines.append("")
    if done:
        hb, ht = holm([r["p_boot"] for _, r in done]), holm([r["p_seed"] for _, r in done])
        lines += ["| question | A − B | metric | Δ points [95% CI] | per-seed Δ | p items (Holm) | p seeds (Holm) "
                  "| resolved |", "|---|---|---|---|---|---|---|---|"]
        for ((q, a, b, g, null), r), pb, pt in zip(done, hb, ht):
            ok = pb < 0.05 and pt < 0.05 and (null == 0 or r["diff"] > null)
            lines.append(f"| {q} | {a} − {b} | {g} | {100 * r['diff']:+.1f} [{100 * r['ci95'][0]:+.1f}, "
                         f"{100 * r['ci95'][1]:+.1f}] | {', '.join(f'{100 * d:+.1f}' for d in r['per_seed'])} | "
                         f"{pb:.3g} | {pt:.3g} | {'yes' if ok else 'no'} |")
    lines += ["", "## Secondary (descriptive, no correction)", "",
              "| A − B | metric | Δ points [95% CI] | per-seed Δ |", "|---|---|---|---|"]
    for a, b in dict.fromkeys((t[1], t[2]) for t in FAMILY):
        if paired(a, b):
            for g, cats in GROUPS.items():
                r = compare(*paired(a, b), cats)
                lines.append(f"| {a} − {b} | {g} | {100 * r['diff']:+.1f} [{100 * r['ci95'][0]:+.1f}, "
                             f"{100 * r['ci95'][1]:+.1f}] | {', '.join(f'{100 * d:+.1f}' for d in r['per_seed'])} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfcl", default="runs/bfcl")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    runs = load_all(args.bfcl)
    os.makedirs(args.out, exist_ok=True)
    text = report(runs)
    open(f"{args.out}/summary.md", "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
