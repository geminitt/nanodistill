<div align="center">

# nanodistill

[![CI](https://img.shields.io/github/actions/workflow/status/geminitt/nanodistill/ci.yml?branch=main&label=CI&style=for-the-badge)](https://github.com/geminitt/nanodistill/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/PYTHON-3.12-A19654?style=for-the-badge)](./pixi.toml)
[![License](https://img.shields.io/badge/LICENSE-MIT-6B7F4E?style=for-the-badge)](./LICENSE)

**Distilling tool calling from Qwen3-8B into Qwen3-0.6B: which way of learning from the teacher transfers the
most, and does it survive unseen tools and requests that should not call any tool?**

</div>

---

> **Note:** **This project did not reach its goal.** The aim was a small tool-calling model better than what can
> already be downloaded. No trained student is: the most balanced one, `irrelevance`, beats Qwen3-0.6B by 5.3
> points on the primary metric but handles 16.4 points fewer of the requests that should not call any tool
> ([Results](#results-5-of-7-conditions)). The two comparisons closest to the central question, soft against
> hard targets and filtering the teacher's answers, were never run. No model is released. What remains useful
> are the findings on refusal examples and test-time reasoning, and the evaluation checks and incidents below.

> **Status:** **closed at 5 of 7 conditions** (15 of 21 training runs). `seq_kd` and `logit_kd` were lost to an
> evaluation bug ([Incidents](#incidents), 1) and not retrained: that needs paid compute, and the study stops
> here. The Design and Pre-registered analysis sections were committed in `8c0c0f3` before any student was
> trained and are unchanged below; [Deviations](#deviations-from-the-pre-registration) lists every place the
> study departs from them.

---

## Design

| | |
|---|---|
| Task | Function calling: given tool schemas and a user request, emit the right calls (name and arguments), or none |
| Training prompts | 5,000 examples of [xLAM function-calling 60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) (CC BY 4.0), a fixed random sample. Every xLAM example whose tools share a name with any BFCL function is removed first (6,118 of 60,000), so BFCL's tools stay unseen |
| Irrelevance prompts | 1,000 further xLAM examples with every tool the answer calls removed: no remaining tool fits the request |
| Teacher | Qwen3-8B, bf16, one configuration for all teacher data: greedy direct answers with top-20 log-probabilities; reasoning traces in thinking mode (temperature 0.6, top-p 0.95, top-k 20) |
| Student | Qwen3-0.6B-Base, full fine-tuning in bf16 autocast. Every condition: same start, 400 optimizer steps × 32 sequences, AdamW, peak LR 2e-5 with 20 warmup steps and cosine decay. The seed sets the data order; 3 seeds per condition |
| Format | Exactly the prompt BFCL's Qwen3 handler builds (`tests/test_bfcl_format.py` checks it against the handler) |
| Evaluation | BFCL v4 single-turn categories, AST checking, temperature 0.001. Students without reasoning traces run in Qwen3's non-thinking mode, as trained |
| Baselines | Qwen3-0.6B (instruct) and Qwen3-8B, in thinking and non-thinking mode |

**Conditions** (only what the student learns from differs):

| Condition | The student learns from |
|---|---|
| `sft_gold` | xLAM's execution-verified calls; no teacher |
| `seq_kd` | the teacher's direct answers, all of them |
| `seq_kd_filtered` | the teacher's direct answers whose calls match the verified ones |
| `logit_kd` | the teacher's top-20 distribution on the same sequences as `seq_kd` (soft cross-entropy) |
| `cot_full` | the teacher's reasoning and calls (filtered); the student also reasons at test time |
| `cot_train_only` | `seq_kd_filtered` plus the filtered reasoning traces; the student answers directly at test time |
| `irrelevance` | `seq_kd_filtered` plus the teacher's answers to the irrelevance prompts that call no tool |

On-policy distillation (the student samples, the teacher scores) is the next step if budget remains; it is not
part of the pre-registered comparisons.

---

## Pre-registered analysis

**Metrics.** Primary: accuracy on BFCL's non-live Python AST categories (simple, multiple, parallel,
parallel_multiple; 1,000 items). Secondary: live AST categories, irrelevance (non-live and live), and output
tokens per request.

**Comparisons**, each "A against B":

| Question | Comparison |
|---|---|
| Does the teacher beat the dataset's own answers? | `seq_kd_filtered` vs `sft_gold` |
| Does filtering the teacher's answers help? | `seq_kd_filtered` vs `seq_kd` |
| Do soft targets beat hard targets on the same sequences? | `logit_kd` vs `seq_kd` |
| Do reasoning traces help, at what cost in output tokens? | `cot_full` vs `seq_kd_filtered`; `cot_train_only` vs `seq_kd_filtered` |
| Do refusal examples teach the student not to call? | `irrelevance` vs `seq_kd_filtered` on the irrelevance categories; and on the primary metric, where it must not lose more than 2 points (non-inferiority) |

**Decision rule.** A difference is resolved only when both hold:
1. *Test items:* the paired bootstrap 95% interval (10,000 resamples over items, per-item accuracy averaged over
   the three seeds) excludes 0;
2. *Training runs:* pairing the runs by seed, the three per-seed differences pass a one-sample t test at 5%
   (2 degrees of freedom).

Holm's correction applies across the seven comparisons on the primary metric. Negative and null results are
reported as they come.

**Reported alongside:** how many teacher answers the filters keep, how often each student emits unparseable
output, and the name overlap removed between xLAM and BFCL.

---

## Results (5 of 7 conditions)

Generated by `nanodistill.analysis` into [`results/summary.md`](results/summary.md). Every student was
evaluated on the same local GPU (RTX 1000 Ada, 6 GB, bf16); accuracies are in %, means over 3 seeds.

| Model | Primary | Live AST | Irrelevance | Output tokens (median) |
|---|---:|---:|---:|---:|
| `sft_gold` | 81.4 | 60.5 | 10.3 | 45 |
| `seq_kd_filtered` | 80.9 | 54.1 | 11.7 | 45 |
| `irrelevance` | 81.0 | 53.7 | 65.8 | 45 |
| `cot_train_only` | 82.2 | 56.6 | 4.2 | 45 |
| `cot_full` | 72.9 | 55.3 | 28.8 | 330 |
| *Qwen3-0.6B-Base, untrained (Modal)* | *0.0* | *0.0* | *100* | *2048* |
| *Qwen3-0.6B, non-thinking / thinking (local)* | *75.7 / 78.5* | *50.0 / 57.7* | *82.2 / 80.0* | *44 / 264* |
| *Qwen3-8B teacher, non-thinking / thinking (Modal)* | *93.9 / 92.8* | *77.1 / 79.6* | *78.6 / 78.1* | *45 / 335* |

> **Note:** Baselines (italics) are context, not part of the tests. Qwen3-0.6B-Base never emits a tool call, so it
> "passes" every irrelevance item; irrelevance accuracy must always be read next to the primary metric.

**Pre-registered tests** (Holm's correction over the 5 that could be run):

| Question | A − B | Metric | Δ points [95% CI] | Per-seed Δ | Resolved |
|---|---|---|---|---|---|
| Teacher vs dataset answers | `seq_kd_filtered` − `sft_gold` | primary | −0.5 [−1.9, +0.8] | −1.3, +0.8, −1.1 | no |
| Reasoning traces, also at test time | `cot_full` − `seq_kd_filtered` | primary | −8.0 [−10.2, −5.8] | −7.2, −7.1, −9.8 | yes* |
| Reasoning traces, training only | `cot_train_only` − `seq_kd_filtered` | primary | +1.3 [−0.3, +3.1] | +1.9, +1.2, +0.9 | no |
| Refusal examples teach not calling | `irrelevance` − `seq_kd_filtered` | irrelevance | +54.1 [+51.3, +56.9] | +54.1, +58.0, +50.3 | yes |
| Refusal examples cost ≤ 2 points | `irrelevance` − `seq_kd_filtered` | primary | +0.1 [−1.2, +1.3] | +0.3, −0.6, +0.5 | no** |

\* Resolved with 5 tests; had the two missing tests been run and come out null, the per-seed p-value after
correction over 7 would be 0.071 and it would not be. \*\* The interval lies above −2 points, but three seeds do not give the per-seed
t test enough power after correction.

What the numbers say so far:
- **Refusal examples work and cost nothing measurable**: +54 points on irrelevance, primary unchanged.
- **Reasoning at test time hurts** this student: −8 points on the primary metric at 7× the output tokens.
  About 11% of its outputs never close the think block before BFCL's 4,096-token budget (greedy decoding
  loops), but those account for only ~10% of its primary errors; most errors are wrong calls after reasoning.
- **Reasoning traces used only in training**: +1.3 points, not resolved. Note that equal steps × sequences
  means `cot_train_only` trains on 8× more target tokens than `seq_kd_filtered` (1.67M vs 0.20M), so even a
  resolved gain could not be credited to the reasoning content alone.
- **Teacher answers vs the dataset's own**: no difference on the primary metric; the dataset answers do
  better on live AST (−6.4 points, secondary, uncorrected).

**Against the ready-made Qwen3-0.6B** (descriptive; the baseline has one run): no student is better on every
metric at once. The most balanced student, `irrelevance`, beats Qwen3-0.6B in non-thinking mode by 5.3 points
on the primary metric and 3.7 on live AST at the same output length, but handles 16.4 points fewer of the
requests that should not call any tool (65.8 against 82.2). An agent built on it would make more wrong calls,
which usually cost more than missed ones, so these students do not replace Qwen3-0.6B. What they do show is
how far 400 steps on 5,000 examples move the base model: from 0 to 81 on the primary metric.

**Reported alongside** (pre-registered): the direct-answer filter keeps 3,917 of 5,000 teacher answers
(78.3%), the reasoning filter 3,994 of 5,000 (79.9%); the teacher declines to call a tool on 925 of the 1,000
irrelevance prompts; 6,118 of 60,000 xLAM examples were removed for sharing a tool name with BFCL. Unparseable
outputs per student: 0–5 of 3,641 (table in `results/summary.md`).

---

## Deviations from the pre-registration

| Pre-registered | What was done | Why |
|---|---|---|
| Holm's correction "across the seven comparisons on the primary metric" | One family of 7 tests, each on its own metric (6 differences against 0, plus non-inferiority against −2 points), Holm over all 7; other metrics are descriptive | The seventh comparison is the irrelevance-metric test; the sentence cannot be read literally |
| Baselines: Qwen3-0.6B and Qwen3-8B | Qwen3-0.6B-Base added | Shows what distillation adds from the students' starting point |
| Evaluation settings unspecified beyond BFCL | Students and 0.6B baselines on a local 6 GB GPU with context capped at 12,288 tokens and Qwen3-0.6B's generation settings; Qwen3-8B and Base on Modal at BFCL's default 40,960 | The cap fits 6 GB and changes nothing: the longest prompt is 6,178 tokens, so every item keeps BFCL's 4,096-token budget. One generation config for every 0.6B model (Base's own would stop output at 2,048 tokens) |
| 7 conditions, 7 tests | 5 conditions; Holm's correction over the 5 tests that could be run | `seq_kd` and `logit_kd` were overwritten (Incidents, 1); retraining them needs paid compute, and the study was closed without them |
| On-policy distillation as the next step if budget remains | Not done | The study was closed at 5 conditions |

---

## Incidents

1. **Evaluation overwrote 15 trained students.** The first evaluation copy linked the student weights and then
   copied a Hugging Face snapshot folder whose cached `model.safetensors` went through the link: all 15
   non-reasoning students became byte-identical to Qwen3-0.6B, and their first scores were Qwen3-0.6B's.
   Found from near-identical scores and confirmed by hashes; the weights and scores were deleted and
   `seq_kd_filtered`, `irrelevance` and `sft_gold` retrained (their loss curves track the first runs' logs).
   The copy now never links and never writes into a model folder (`tests/test_eval_copy.py`); hashes are
   checked after every evaluation.
2. **Wrong RoPE base at evaluation.** Students are saved by transformers 5, which keeps the RoPE base in
   `rope_parameters`; vLLM 0.8.5 (transformers 4.51, pinned by BFCL) reads only `rope_theta` and silently
   used 10,000 instead of 1,000,000. The evaluation copy now writes both. On 60 BFCL prompts the fixed setup
   reproduces transformers 5 greedy outputs 59/60 times (the unfixed one 2/60; vLLM 0.30 reading the saved
   model directly 57/60).
3. **Corrupted downloads.** Large files downloaded from the Modal volume came back with runs of a few MiB
   zeroed or cut short. `nanodistill.fetch` downloads three copies at full size and rebuilds each 64 KiB
   block from a copy where it is not all zero; trained weights never contain an all-zero block.

---

## Measurement checks

- Per-item correctness read from BFCL's files equals BFCL's own correct counts in all 286 category files.
- Weights evaluated = weights trained: every student's loss on a sample of its own training sequences is
  0.0004–0.30 nats/token (Base: 1.35), and its hash is the same before and after evaluation.
- Training and evaluation tokenize prompts identically (transformers 5.17 vs 4.51, 60/60 prompts).
- Evaluation noise: the same student evaluated twice locally differs on 21 of 3,641 items (±0.2 points);
  Qwen3-0.6B local vs Modal differs on 1% of items without thinking and 8% with thinking (no direction:
  sign-test p = 0.29 on the primary metric). All students are therefore evaluated on one machine.

---

## Layout

```
src/nanodistill/  data (format), teacher and train (Modal), fetch (verified download), bfcl_local
                  (local BFCL), bfcl_runner (BFCL CLI with the non-thinking handler), bfcl_eval_modal
                  (BFCL on Modal, for Qwen3-8B), analysis (pre-registered tests)
data/             BFCL function names used to remove overlapping xLAM examples
results/          summary.md, written by nanodistill.analysis
tests/            pytest; BFCL-dependent tests run in the eval environment
```

Environments: `pixi run test` (default, CPU) and `pixi run -e eval pytest -q tests` (BFCL installed).
Compute runs on [Modal](https://modal.com); `pixi run modal setup` authenticates once; a Modal secret named
`huggingface-secret` holds a read-only Hugging Face token (xLAM is gated).

---

## Reproduce

```bash
pixi run modal run --detach src/nanodistill/teacher.py --mode direct        # and --mode cot
pixi run modal run --detach src/nanodistill/train.py --conditions all --seeds 0,1,2
pixi run python -m nanodistill.fetch seq_kd_filtered_s0 ...                  # students -> runs/students/
pixi run -e eval python -m nanodistill.bfcl_local --label seq_kd_filtered_s0 \
    --model runs/students/seq_kd_filtered_s0 --mode direct                   # --mode think for cot_full
pixi run modal run src/nanodistill/bfcl_eval_modal.py --models baselines    # Qwen3-8B does not fit 6 GB
pixi run python -m nanodistill.analysis --bfcl runs/bfcl --out results
```
