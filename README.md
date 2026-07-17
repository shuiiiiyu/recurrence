# State Reccurence

Minimal experimental code for a standard causal Transformer baseline plus two depth-recurrence cells in the recurrent-transformer taxonomy:

- `baseline`: Standard causal Transformer with no depth recurrence.
- `ratio1`: Depth recurrence with input-token / recurrence-step ratio = 1. The previous token's deep state is injected into the current token's shallow layer.
- `ratiolt1`: Depth recurrence with ratio < 1. Each external token is held fixed while the middle/deep stack is looped multiple times. Then the previous token's deep state is injected into the current token's shallow layer.

## Layout

```text
configs/default.yaml              # readable default experiment settings
scripts/smoke_test.sh             # quick CPU sanity check for all models
scripts/train_baseline.sh        # standard Transformer baseline entrypoint
scripts/train_ratio1.sh           # ratio = 1 entrypoint
scripts/train_ratiolt1.sh         # ratio < 1 entrypoint
recurrence_model/config.py    # main parameters and dataclass config
recurrence_model/data.py      # synthetic state-tracking dataset
recurrence_model/models.py    # baseline and depth-recurrence model variants
recurrence_model/training.py  # training and evaluation loop
scripts/train.py         # thin CLI wrapper
```

## Task Format

The training code now uses token-level labels:

```text
input_ids: [seq_len]
labels:    [seq_len]
```

Label `0` means ignore this position in the loss. The model returns logits for every input position:

```text
logits: [batch, seq_len, vocab_size]
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
input:  story permutation tokens + OUT slots
label:  ignore story positions, final permutation values on OUT slots
```

For example, S3 has 100 story tokens and 3 OUT slots, so `seq_len = 103`. The debug script samples a tiny subset; the full script uses the complete selected subset.

## Data

Included in this repository:

- `data/raw/grade_school_math` - 14 MB
- `data/raw/babi` - 124 MB

Kept outside Git because they are too large for a normal GitHub repository:

- `/data/shencanyu/data/raw/permutation` - 2.0 GB, many small JSON files
- `/data/shencanyu/data/raw/sudoku-extreme` - 822 MB, includes Arrow shards over 100 MB

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
scripts/train_sudoku_debug.sh
```
Override parameters from the command line, for example:

```bash
scripts/train_ratio1.sh --num_layers 24 --d_model 384 --num_heads 6
```
Datasets, model weights, caches, checkpoints, logs, and local environments stay outside Git.
