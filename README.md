# State Recurrence

This repository contains experimental code for recurrent transformer taxonomy experiments.

## Current Status

The repository currently implements two recurrence axes:

- Depth recurrence: implemented with Transformer blocks.
- Step recurrence: implemented with Mamba blocks.

Training and evaluation are currently running on two state-tracking benchmark families:

- Permutation: S3 and S5 tasks with sequence lengths 50 and 100. Each setting uses 10k-scale data.
- bAbI: 20 tasks with different difficulty levels. Each task uses the en-10k split.

Other recurrence axes, datasets, and experiment variants are still under development.

## Results

Current result summaries are stored under `results/`:

- `depth_permutation_results.csv`
- `depth_babi_results.csv`
- `step_permutation_results.csv`
