# Dataset Distillation for Memorized Data: Soft Labels can Leak Held-Out Teacher Knowledge

> Freya Behrens, Lenka Zdeborová

Code accompanying the paper.

## Usage

All experiments use the basic code available at `mod_addition_experiment.py`, `logistic_regression_experiment.py` or `experiment.py`.

For reproducibility we record most experiments to a wandb database, along with the random seeds that were used to create them. One can use the experiment (`.py`) files to re-run them, and the plotting notebooks (`.ipynb`) to create the plots for the figures in the corresponding names of the files.

For the visual examples `fig_1` and `fig_7` the seeds are configured in the plotting file directly.
For logistic regression in `fig_3` we save and provide the results in the repository.

## Resources

All experiments can be run both on cpu or gpu - for multinomial logistic regression ththe cpu is probably be faster than a gpu.

The most computational intensive were the phase diagrams Fig. 4a) with 13 compute days, and Fig. 6a) with roughly 10 compute days on a single GPU, an NVIDIA RTX A5000.

Since many of the experiments are run for different seeds to obtain error bars, running the experiments once is a factor 5 faster than running all. Since experiments do not require large resources they can be parallelized easily.
