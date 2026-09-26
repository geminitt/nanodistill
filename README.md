# nanodistill

**Distilling tool calling from Qwen3-8B into Qwen3-0.6B: which way of learning from the teacher transfers the
most, and does it survive unseen tools and requests that should not call any tool?**

Status: pre-registration. The design and analysis below are committed before any student is trained.

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

## Layout

```
src/nanodistill/  data (format), teacher (Modal), train (Modal), bfcl_eval_modal (Modal)
data/             BFCL function names used to remove overlapping xLAM examples
tests/            pytest; the format test runs in the eval environment
```

Environments: `pixi run test` (default, CPU) and `pixi run -e eval pytest -q tests` (BFCL installed).
Compute runs on [Modal](https://modal.com); `pixi run modal setup` authenticates once.
