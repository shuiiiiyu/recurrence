# State Recurrence

Minimal experimental code for a standard causal Transformer baseline plus two depth-recurrence cells in the recurrent-transformer taxonomy.

## Current Status

This repository currently implements only:

- `baseline`: Standard causal Transformer with no depth recurrence.
- `ratio1`: Depth recurrence with input-token / recurrence-step ratio = 1. The previous token's deep state is injected into the current token's shallow layer.
- `ratiolt1`: Depth recurrence with ratio < 1. Each external token is held fixed while the middle/deep stack is looped multiple times. Then the previous token's deep state is injected into the current token's shallow layer.

Other recurrent-transformer taxonomy cells, additional datasets, and larger experiment tooling are still under development.

## Layout

```text
configs/default.yaml              # readable default experiment settings
scripts/smoke_test.sh             # quick CPU sanity check for all models
scripts/train_baseline.sh         # standard Transformer baseline entrypoint
scripts/train_ratio1.sh           # ratio = 1 entrypoint
scripts/train_ratiolt1.sh         # ratio < 1 entrypoint
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
