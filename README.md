# HT Schatten Experiments

Code for reproducing the experiments of the paper _Free Heavy-Tailed Lunch for Muon: A Theoretical Justification of Empirical Success_.

## Setup

```bash
python -m pip install -e .
```

## Training

```bash
accelerate launch --num_processes 2 \
  -m schatten_experiments.train \
  --config configs/schatten_run.yaml \
  --overwrite learning_rate=0.0075 schatten_r=inf seed=0 \
  --output-dir [output_dir]/learning_rate0.0075_schatten_rinf_seed0
```

Set `--num_processes` to the number of processes you want for each job. The provided `schatten_run.yaml` gradient accumulation is calibrated for 2 processes; adjust `gradient_accumulation_steps` if you change the process/GPU count.

## Noise Statistics And Figures

```bash
accelerate launch --num_processes 2 \
  -m schatten_experiments.stats.examine_noise \
  --config configs/noise_ratios.yaml \
  --run-dir [output_dir]/learning_rate0.0075_schatten_rinf_seed0
```

```bash
python -m schatten_experiments.plots.noise_ratios \
  --run-dir [output_dir]/learning_rate0.0075_schatten_rinf_seed0 \
  --output-dir [output_dir]/figures/noise_ratios
```

The plotter writes the normalized entrywise and Schatten noise-ratio snapshots used for the paper, including embedding and `lm_head` layers by default.

## Notes

- Put Hugging Face caches on local/scratch storage with `HF_HOME` and, if needed, `HF_DATASETS_CACHE`.
- Set `mixed_precision: "no"` in the YAML configs if bfloat16 is unsupported.
- Outputs include `experiment_config.yaml`, `experiment_config.json`, and `all_results.json`; noise stats are written to `stats/<checkpoint>/opt_stats.json`.

## Citation

If you use this code, please cite the accompanying paper:

~~~bibtex
@article{FreeHTLunch2026huebler,
  title={Free Heavy-Tailed Lunch for Muon: A Theoretical Justification of Empirical Success},
  author={H{\"u}bler, Florian and Pethick, Thomas and Sra, Suvrit},
  journal={arXiv preprint arXiv:2606.14560},
  year={2026}
}
~~~
