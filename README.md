# State Recurrence

This repository contains experimental code for recurrent transformer taxonomy experiments.

## Current Scope

The repository currently implements depth recurrence, step recurrence, and depth+step recurrence models for state-tracking experiments.

Step recurrence models directly reuse code or core implementations from existing papers. Depth recurrence ratio=1 and ratio<1 are implemented in this repository based on the architectural descriptions in the taxonomy paper. Other depth-recurrence variants are adapted from existing architecture code. Depth+step recurrence variants are also implemented and evaluated as a separate family.

## Self-Implemented Depth Recurrence

`depthRatio1` processes one new token per recurrence step. At each step, the model recomputes the visible causal prefix, carries each token's previous source-layer hidden state, and feeds that same-token state back into a shallower target layer during the next prefix update.

`depthRatiolt1` uses the same prefix-recurrent feedback mechanism, but performs multiple recurrent updates for each newly introduced token before advancing to the next input token.

## Training And Evaluation

Training and evaluation are currently running on two state-tracking benchmark families:

Permutation: S3 and S5 tasks with sequence lengths 50 and 100. Each setting uses 100k-scale data.

bAbI: 20 tasks with different difficulty levels. Each task uses the en-10k split.
