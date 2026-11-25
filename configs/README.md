# Experiment configs

Choose the experiment you want to run:

1. **Predict a new batch.** Tune on the training batches, refit the chosen model,
   then test on a batch left out of training and tuning. Use
   [out_of_batch/rxrx1.yaml](out_of_batch/rxrx1.yaml) or
   [out_of_batch/replogle.yaml](out_of_batch/replogle.yaml).
2. **Recover a mean from sparse responses.** Keep responses from only a few
   batches per perturbation. Tune on those responses, then compare the estimated
   mean with the full observed mean. Use
   [sparse_recovery/rxrx1.yaml](sparse_recovery/rxrx1.yaml).
   Replogle sparse recovery has no runner yet.

Run these commands from the project root. Each script loads its experiment's
config. Set `name` to choose the run's output folder.

```bash
uv run python scripts/tune_rxrx1_transfer.py name=rxrx1_tuned neural=hybrid
uv run python scripts/tune_replogle_transfer.py name=replogle_tuned neural=hybrid
uv run python scripts/evaluate_rxrx1_sparse_recovery.py name=rxrx1_sparse neural=hybrid
```

Change a setting in the file or on the command line. For example:

```bash
uv run python scripts/tune_rxrx1_transfer.py \
  name=rxrx1_shared neural=shared_2nn n_configs=10 neural.device=cpu
```

Use `neural=per_x`, `neural=shared_2nn` or `neural=hybrid` to choose the neural
model.

Out-of-batch tuning uses `bridges=[neural]` by default. Add the kernel bridge with
`'bridges=[kernel,neural]'`. Sparse recovery uses both by default. Out-of-batch
runs always include linear baselines; sparse recovery includes them only when
you set `include_linear=true`.

Print all settings before running an experiment:

```bash
uv run python scripts/tune_rxrx1_transfer.py name=check neural=hybrid --cfg job --resolve
```

This command loads no data and trains no models. Command-line values take
precedence over file settings. During tuning, each trial replaces the model
settings it tests.

The remaining files hold shared settings or support other runs:

- [common.yaml](common.yaml) selects model defaults and links the seed, count
  weights and input standardisation settings.
- [search_spaces/rxrx1.yaml](search_spaces/rxrx1.yaml) and
  [search_spaces/replogle.yaml](search_spaces/replogle.yaml) list the values to
  try and the training steps at which to score neural models.
- `out_of_batch/*_fixed.yaml` holds settings for runs without tuning. The
  `train_rxrx1_transfer.py` and `train_replogle_transfer.py` scripts use these.
- [preprocessing/embed_rxrx1.yaml](preprocessing/embed_rxrx1.yaml) controls image
  embedding.

Each experiment loads shared files directly. It does not inherit settings from
another experiment. RxRx1 uses site normalisation for out-of-batch prediction
and leaves it off for sparse recovery.

The script commands are unchanged. If you pass `--config-name`, use the new
path, such as `--config-name out_of_batch/rxrx1`. The old flat names are gone.
Sparse runs also check source-code hashes before resuming; use the original
code to resume a run started before these script changes.
