# State Recurrence

Minimal experimental code for a standard causal Transformer baseline plus depth-recurrence cells in the recurrent-transformer taxonomy.

## Current Status

This repository currently implements only:

- `baseline`: Standard causal Transformer with no depth recurrence.
- `ratio1`: Depth recurrence with input-token / recurrence-step ratio = 1. The previous token's deep state is injected into the current token's shallow layer.
- `ratiolt1`: Depth recurrence with ratio < 1. Each external token is held fixed while the middle/deep stack is looped multiple times. Then the previous token's deep state is injected into the current token's shallow layer.
- `ratiogt1`: Depth recurrence with ratio > 1. The full sequence is processed in parallel with causal attention, while a middle layer range is looped multiple times in depth. There is no cross-token deep-to-shallow feedback.

Other recurrent-transformer taxonomy cells, additional datasets, and larger experiment tooling are still under development.

## Layout

```text
configs/default.yaml              # readable default experiment settings
scripts/smoke_test.sh             # quick CPU sanity check for all models
scripts/train_baseline.sh         # standard Transformer baseline entrypoint
scripts/train_ratio1.sh           # ratio = 1 entrypoint
scripts/train_ratiolt1.sh         # ratio < 1 entrypoint
scripts/train_ratiogt1.sh         # ratio > 1 entrypoint
recurrence_model/config.py        # main parameters and dataclass config
recurrence_model/data.py          # dataset loaders and tokenization
recurrence_model/models.py        # baseline and depth-recurrence model variants
recurrence_model/training.py      # training and evaluation loop
scripts/train.py                  # thin CLI wrapper
```
### Sudoku

Sudoku follows the TinyRecursiveModels-style format:

```text
input:  81 puzzle cells, '.' encoded as blank
label:  81 solution digits
```

Encoding:

```text
pad = 0
blank = 1
digit d = d + 1
```

### Permutation

Permutation examples use the raw JSON files under `/data/shencanyu/data/raw/permutation`:

```text
input:  story permutation tokens
label:  cumulative permutation state after each input token
```

For example, S3 with 100 story tokens has `seq_len = 100`. Each label is one permutation-state class such as `213`, aligned to the input token that produced that state. This is the aligned chain-of-thought (ACoT) format. The debug script samples a tiny subset; the full script uses the complete selected subset.

Permutation metrics include:

```text
test/token_acc  # per-step state accuracy on the test split
test/final_acc  # final composed permutation accuracy on the test split; the main task metric
test/exact_acc  # all intermediate states correct for the whole sequence; strict auxiliary metric
```

Training uses the dataset's `train` split and evaluation uses the dataset's `test` split. Use `--max_train_samples` and `--max_test_samples` to keep their sizes at a comparable ratio for quick runs.

## Data

Included in this repository:

- `data/raw/grade_school_math` - 14 MB
- `data/raw/babi` - 124 MB

Kept outside Git because they are too large for a normal GitHub repository:

- `/data/shencanyu/data/raw/permutation` - 2.0 GB, many small JSON files
- `/data/shencanyu/data/raw/sudoku-extreme` - 822 MB, includes Arrow shards over 100 MB


## Epochs And Steps

Training normally uses `--epochs`. The code converts epochs to optimizer steps as:

```text
steps_per_epoch = ceil(num_train_samples / batch_size)
total_steps = ceil(epochs * steps_per_epoch)
```

You can still pass `--max_steps` to manually override the computed total steps.

## Smoke Test

```bash
scripts/smoke_test.sh
```
## Main Entrypoints
```bash
scripts/train_baseline.sh
scripts/train_ratio1.sh
scripts/train_ratiolt1.sh
scripts/train_permutation_full.sh
scripts/train_permutation_debug.sh
scripts/overfit_permutation_train_eval.py
scripts/train_sudoku_debug.sh
```
Override parameters from the command line, for example:

```bash
scripts/train_ratio1.sh --num_layers 24 --d_model 384 --num_heads 6 --epochs 1
```

Use `scripts/overfit_permutation_train_eval.py` for tiny-sample sanity checks where the same permutation train subset is used for training and evaluation.

Datasets, model weights, caches, checkpoints, logs, and local environments stay outside Git.
