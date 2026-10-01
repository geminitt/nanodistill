# Results

## Accuracy by model (mean over seeds; baselines have one run)

| model | seeds | primary | live AST | irrelevance | output tokens (median) | unparseable |
|---|---|---|---|---|---|---|
| baseline:qwen3-0.6b-base_direct | 1 | 0.0 | 0.0 | 100.0 | 2048 | 16 |
| baseline:qwen3-0.6b-local_direct | 1 | 75.7 | 50.0 | 82.2 | 44 | 3 |
| baseline:qwen3-0.6b-local_think | 1 | 78.5 | 57.7 | 80.0 | 264 | 6 |
| baseline:qwen3-0.6b_direct | 1 | 75.5 | 50.3 | 82.4 | 44 | 3 |
| baseline:qwen3-0.6b_think | 1 | 79.5 | 59.2 | 80.0 | 268 | 5 |
| baseline:qwen3-8b_direct | 1 | 93.9 | 77.1 | 78.6 | 45 | 2 |
| baseline:qwen3-8b_think | 1 | 92.8 | 79.6 | 78.1 | 335 | 4 |
| cot_full | 3 | 72.9 | 55.3 | 28.8 | 330 | 4 |
| cot_train_only | 3 | 82.2 | 56.6 | 4.2 | 45 | 1 |
| irrelevance | 3 | 81.0 | 53.7 | 65.8 | 45 | 5 |
| seq_kd_filtered | 3 | 80.9 | 54.1 | 11.7 | 45 | 0 |
| sft_gold | 3 | 81.4 | 60.5 | 10.3 | 45 | 1 |

## Pre-registered tests

Resolved = the item-bootstrap p-value **and** the per-seed t-test p-value are both below 0.05 after one Holm correction across the tests in the table (and, for non-inferiority, the difference is above -2 points).

Only 5 of the 7 tests have both conditions evaluated (the study was closed without the others); the correction below covers those only.

| question | A − B | metric | Δ points [95% CI] | per-seed Δ | p items (Holm) | p seeds (Holm) | resolved |
|---|---|---|---|---|---|---|---|
| teacher vs dataset answers | seq_kd_filtered − sft_gold | primary | -0.5 [-1.9, +0.8] | -1.3, +0.8, -1.1 | 0.443 | 0.509 | no |
| reasoning traces, also at test time | cot_full − seq_kd_filtered | primary | -8.0 [-10.2, -5.8] | -7.2, -7.1, -9.8 | 0 | 0.0476 | yes |
| reasoning traces, training only | cot_train_only − seq_kd_filtered | primary | +1.3 [-0.3, +3.1] | +1.9, +1.2, +0.9 | 0.239 | 0.092 | no |
| refusal examples teach not calling | irrelevance − seq_kd_filtered | irrelevance | +54.1 [+51.3, +56.9] | +54.1, +58.0, +50.3 | 0 | 0.0085 | yes |
| refusal examples cost at most 2 points (non-inferiority) | irrelevance − seq_kd_filtered | primary | +0.1 [-1.2, +1.3] | +0.3, -0.6, +0.5 | 0.0042 | 0.0773 | no |

## Secondary (descriptive, no correction)

| A − B | metric | Δ points [95% CI] | per-seed Δ |
|---|---|---|---|
| seq_kd_filtered − sft_gold | primary | -0.5 [-1.9, +0.8] | -1.3, +0.8, -1.1 |
| seq_kd_filtered − sft_gold | live AST | -6.4 [-8.0, -4.9] | -7.2, -7.0, -5.1 |
| seq_kd_filtered − sft_gold | irrelevance | +1.3 [+0.4, +2.3] | +1.5, -0.3, +2.8 |
| cot_full − seq_kd_filtered | primary | -8.0 [-10.2, -5.8] | -7.2, -7.1, -9.8 |
| cot_full − seq_kd_filtered | live AST | +1.2 [-0.8, +3.2] | +4.1, +0.8, -1.3 |
| cot_full − seq_kd_filtered | irrelevance | +17.1 [+14.8, +19.4] | +16.9, +18.0, +16.5 |
| cot_train_only − seq_kd_filtered | primary | +1.3 [-0.3, +3.1] | +1.9, +1.2, +0.9 |
| cot_train_only − seq_kd_filtered | live AST | +2.5 [+0.8, +4.2] | +5.6, +3.1, -1.3 |
| cot_train_only − seq_kd_filtered | irrelevance | -7.5 [-8.9, -6.1] | -10.7, -5.6, -6.1 |
| irrelevance − seq_kd_filtered | primary | +0.1 [-1.2, +1.3] | +0.3, -0.6, +0.5 |
| irrelevance − seq_kd_filtered | live AST | -0.4 [-1.8, +1.0] | +0.1, -0.8, -0.4 |
| irrelevance − seq_kd_filtered | irrelevance | +54.1 [+51.3, +56.9] | +54.1, +58.0, +50.3 |
