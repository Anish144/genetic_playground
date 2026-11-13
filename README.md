# Introduction

Out-of-batch prediction estimates a known perturbation's response in a new experimental batch. We train on other batches and use control samples from the new batch as a proxy for its conditions. The new batch's perturbed samples are used only for evaluation. The targets are mean gene expression for Replogle and mean image embeddings for RxRx1.

# Datasets

## Replogle

We use the scPerturb version of the Replogle 2022 K562 essential-gene CRISPRi screen. The full-data counts below are recorded in the exploration notebooks and experiment logs.

| Item | Count or definition |
| --- | --- |
| Cells | 310,385: 299,694 perturbed and 10,691 controls |
| Measured genes | 8,563 expression features |
| Perturbations | 2,057 gene-target labels, plus `control` (2,058 labels total) |
| Batches | 48, defined by `adata.obs["batch"]` |
| Cell type | K562 |

Measured genes are output features; targeted genes are perturbation labels. These sets differ. Perturbations have unequal cell counts and do not all occur in every batch.

`data/data_loader.py` downloads the [scPerturb H5AD](https://exampledata.scverse.org/pertpy/replogle_2022_k562_essential.h5ad) and applies these steps in order:

1. Remove cells with fewer than 100 total counts.
2. Remove genes detected in fewer than 5 cells.
3. Normalise each cell to 4,000 total counts, then apply `log1p`.
4. Keep single-perturbation cells (`nperts == 1`) and controls.
5. Save `data/replogle_k562_preprocessed.h5ad` and reuse it on later runs.

The current pipeline keeps all 8,563 retained genes. Older runs used 5,000 highly variable genes. The cache is not versioned: move an old cache aside before rerunning if it contains the older preprocessing.

## RxRx1

RxRx1 contains 125,510 microscopy sites, each with six 512 × 512, 8-bit channels stored as separate PNGs. There are two sites per well. These are images of cell populations; the pipeline does not segment or count individual cells. See the [dataset documentation](https://github.com/recursionpharma/rxrx-datasets/blob/trunk/rxrx1/README.md).

A **batch is one experiment**, such as `HUVEC-01`. Each experiment contains one cell type and four 384-well plates, with 308 usable wells per plate. Plates, sites and crops stay within their experiment when splitting data. There are **51 batches**: 24 HUVEC, 11 HEPG2, 11 RPE and 5 U2OS. The design has **1,108 treatment siRNAs**, **30 positive-control siRNAs**, and untreated negative controls (`EMPTY`). Each treatment appears once per experiment; each positive control appears on every plate. Some intended treatments are missing or relabelled `EMPTY`. See the [RxRx1 paper](https://arxiv.org/html/2301.05768v1#S2).

A perturbation is the **(cell type, siRNA) tuple**. With `cell_type=null`, the code labels it `<cell_type>|<siRNA>`. The full design gives 4,432 treatment tuples and 120 positive-control tuples. With the default `cell_type=HUVEC`, cell type is fixed and labels contain only the siRNA. Negative controls share the label `control`, but their proxy distributions remain separate by batch.

Preprocessing in `scripts/embed_rxrx1.py` is:

1. **Crop:** split each site into a regular, non-overlapping `grid × grid`. The default `grid=4` gives 16 crops of 128 × 128 pixels per site. `grid=2` gives four 256 × 256 crops. Keep all crops, including empty or low-intensity ones.
2. **Standardise images:** divide pixel values by 255, then apply instance normalisation independently to each channel of each crop (approximately zero mean and unit variance, with numerical epsilon). With `standardise_sites=true`, normalise each full site before cropping instead.
3. **Embed:** use the frozen [OpenPhenom CA-MAE ViT-S/16 encoder](https://huggingface.co/recursionpharma/OpenPhenom). Keep all six channels and the crop's native resolution. Rebuild its fixed positional embeddings for the crop size. Drop the class token and average patch tokens.
4. **Save:** each crop becomes one **384-dimensional vector**. Store vectors and batch, cell type, siRNA, well, site and crop labels in H5AD. There is no PCA, embedding batch correction or fine-tuning of OpenPhenom.

For all 125,510 sites, the final matrix is **2,008,160 × 384** with `grid=4`, or **502,040 × 384** with `grid=2`. Its path is `data/rxrx1_phenom_<grid>x<grid>[_sitenorm].h5ad`.

The local checkout currently contains only a smoke subset: 88 sites from four experiments (HEPG2-01/02 and HUVEC-01/02), giving **1,408 × 384** embeddings. Older full runs used a removed low-intensity filter and recorded 497,208 crops at `grid=2`; those counts do not describe the current pipeline.

# Methods

Let $x$ be a perturbation, $e$ a batch and $g$ an output feature (gene or embedding coordinate). Let $\bar y_{xe}$ be the mean response, $n_{xe}$ its cell/crop count, and $w_{ej}$ a control sample. Linear models use the control mean $W_e$; bridges use the full control distribution.

**Input standardisation.** By default (`standardise_w=true`), every continuous proxy coordinate uses $\tilde w_g=(w_g-a_g)/s_g$. The mean $a_g$ and population standard deviation $s_g$ are fitted to pooled training controls only, and reused for validation and test controls. Constant coordinates use $s_g=1$. Linear control means use the same transformation. Categorical/one-hot perturbation inputs stay unchanged. Targets and predictions remain in log-expression or Phenom units. Image normalisation above is a separate step.

**Count weighting.** Linear least squares always uses cell/crop counts. For bridges, $c_{xe}=n_{xe}$ by default; `weight_by_cells=false` sets $c_{xe}=1$ for observed pairs. **If perturbation $x$ is absent from batch $e$, its weight is zero.** Missing responses are omitted, not treated as observed zeros. Let $N=\sum_{xe}c_{xe}$. Bridge ridge penalties use $\lambda N$, where $N$ covers all perturbations in the current training split. Each batch's control embedding is an unweighted mean over its proxy samples.

## 1. Linear

Each feature has a perturbation intercept $\gamma_{xg}$ and a batch slope $b_g$ shared across perturbations:

$$
\hat y_{xeg}=\gamma_{xg}+b_g W_{eg},\qquad
\min_{\gamma,b}\sum_{x,e}n_{xe}\sum_g(\bar y_{xeg}-\gamma_{xg}-b_gW_{eg})^2.
$$

Define the training means $\bar y_{xg}=\sum_e n_{xe}\bar y_{xeg}/\sum_e n_{xe}$ and $\bar W_{xg}=\sum_e n_{xe}W_{eg}/\sum_e n_{xe}$. The three settings predict a new batch $e^*$ as follows:

| Setting | Prediction |
| --- | --- |
| Pooled (`b=0`) | $\hat y_{xe^*g}=\bar y_{xg}$ |
| Shift (`b=1`) | $\hat y_{xe^*g}=\bar y_{xg}-\bar W_{xg}+W_{e^*g}$ |
| Transfer (fit $b$) | $\hat y_{xe^*g}=\bar y_{xg}+\hat b_g(W_{e^*g}-\bar W_{xg})$ |

The fitted slope is the within-perturbation least-squares slope:

$$
\hat b_g=
\frac{\sum_{x,e}n_{xe}(W_{eg}-\bar W_{xg})(\bar y_{xeg}-\bar y_{xg})}
{\sum_{x,e}n_{xe}(W_{eg}-\bar W_{xg})^2}.
$$

These equations use original units. The implementation standardises inputs and converts fixed slopes accordingly, so `b=1` still means a unit shift in original units. Each output uses only its matching control coordinate.

## 2. Kernel bridge

The perturbation feature map is the delta/one-hot basis: $\phi(x)=\delta_x$, so $k_x(x,x')=\mathbf{1}[x=x']$. For controls, use an RBF kernel on standardised inputs:

$$
k_w(w,w')=\exp\!\left(-\frac{\|\tilde w-\tilde w'\|^2}{2\ell^2}\right),\qquad
\mu_e=\frac{1}{m_e}\sum_{j=1}^{m_e}\psi(\tilde w_{ej}).
$$

Here $m_e$ is the number of proxy samples and $\psi$ is the RBF feature map. Its batch similarity is $M_{ee'}=\langle\mu_e,\mu_{e'}\rangle$, computed by averaging kernel values over all pairs of controls. The lengthscale is the median pairwise distance of at most 2,000 pooled training controls, multiplied by `lengthscale_scale`.

The bridge predicts $\hat y_{xeg}=\langle h_g,\phi(x)\otimes\mu_e\rangle$. The delta map gives a separate ridge fit per perturbation, with shared batch similarities. On the batches where $x$ occurs:

$$
\alpha_x=(M_x+\lambda N\,\mathrm{diag}(1/c_{xe}))^{-1}\bar Y_x,
\qquad
\hat y_{xe^*}=M_{e^*,x}\alpha_x.
$$

$\bar Y_x$ contains the observed batch means; $M_{e^*,x}$ compares new-batch controls with those training batches. There is no fitted response sharing across perturbations.

## 3. Neural bridge

Both versions learn a control network $\psi$ and use $\mu_e=m_e^{-1}\sum_j\psi(\tilde w_{ej})$.

- **Learned $\phi(x)$:** an MLP maps one-hot perturbations to $p$ features. For each output, $\hat y_{xeg}=\phi(x)^T A_g\mu_e$, with $A_g\in\mathbb R^{p\times q}$. Both networks and the response coefficients are shared across perturbations.
- **Delta $\phi(x)=\delta_x$ (per $x$):** keep the one-hot input and fit a separate $A_x\in\mathbb R^{q\times G}$ for each perturbation: $\hat y_{xe}=A_x^T\mu_e$. Only $\psi$ is shared. Select this with `neural.per_perturbation=true`; `n_features` and `width_x` are unused.

For fixed networks, solve the response coefficients exactly by weighted ridge regression:

$$
\min_A\sum_{x,e}c_{xe}\|\bar y_{xe}-\hat y_{xe}\|_2^2
+\lambda N\|A\|_F^2.
$$

For the per-$x$ version, the penalty sums $\|A_x\|_F^2$ over perturbations, still using the global $N$. Both versions use full-batch AdamW to learn the networks through the ridge solve. MLPs have two SwiGLU hidden layers by default, residual connections where widths match, and linear outputs. Networks run in float32; ridge algebra uses float64.

By default, network training uses five-fold cross-fitting over training batches: at every step, randomly split batches, fit coefficients on the other folds, and score each held-out fold. The loss averages squared error over output features and weights pairs by $c_{xe}$. Final coefficients use all training batches. `cross_fit_folds=0` uses an in-sample loss. Default training runs use 200 steps, learning rate `1e-3`, weight decay `1e-2`, and ridge `1e-6`. Feature sizes are $(p,q)=(32,32)$ for Replogle and $(128,32)$ for RxRx1; tuning overrides these.

# Evaluation

Select distinct validation batches using `seed` (default 0). Hold out each selected batch separately and train on **all other batches**, including those held out in other runs. Defaults are 10 outer evaluations for Replogle and 5 for RxRx1. Score only perturbations present in both the training data and the held-out batch. This evaluates new batches for known perturbations.

**Replogle controls and effects.** Randomly split every batch's controls into two near-equal halves, including training batches. One half supplies proxies and input-scaling statistics. The other half is reserved for evaluation baselines. For each held-out perturbation, subtract that second half's mean $W_e^{\mathrm{eval}}$ from both observed and predicted expression:

$$
\Delta_{xe}=\bar y_{xe}-W_e^{\mathrm{eval}},\qquad
\hat\Delta_{xe}=\hat y_{xe}-W_e^{\mathrm{eval}}.
$$

These differential-expression effects are differences in mean log-normalised expression. Metrics compare these vectors; there is no significance test.

| Dataset | Metrics for each (perturbation, held-out batch) |
| --- | --- |
| Replogle | MSE averaged over all genes; Pearson correlation across all genes; `pearson_top100`, across the 100 genes with largest absolute observed effect (`top_k` is configurable). MSE also includes a **no effect** baseline, $\hat\Delta=0$. |
| RxRx1 | MSE averaged over the 384 coordinates of predicted and observed mean crop embeddings. No control subtraction. |

RxRx1 uses all negative-control crops as proxies, with no half split. Positive controls are training perturbations by default but are never scored. `positive_controls_as_proxies=true` moves them into the proxy pool instead. Only treatment wells are evaluated. Each response mean averages all crops for that perturbation in that experiment.

**Aggregation:** first compute one metric across features for each observed perturbation–batch pair. Then take arithmetic means, without cell/crop weighting:

- **Per batch:** average over its scored perturbations. The overall batch-based result averages these batch scores equally. RxRx1 writes this result to `summary.csv`.
- **Per perturbation (Replogle):** average over selected held-out batches containing that perturbation. The per-perturbation report then averages these scores equally. This can differ from the batch-based result because coverage differs.

Cell counts in score tables describe coverage; they do not weight evaluation averages. `n_training_batches` counts training batches containing each perturbation and is used in coverage plots. Undefined Pearson values are NaN and are skipped by pandas averages.

**Tuning:** nested cross-validation keeps the outer batch out of tuning. Split the remaining batches into three inner folds. Randomly sample bridge settings from the YAML grids and score neural training checkpoints. Select the lowest mean inner MSE: average over perturbations within each batch, then batches within an inner fold, then inner folds equally. Refit on all outer-training batches. The same candidate draw is reused across outer evaluations. When pooling RxRx1 cell types, inner folds distribute experiments by cell type. These inner folds are separate from neural cross-fitting.

# Paths and how to run the experiments

Run commands from the repository root. Use Python 3.12+ and install the locked environment with `uv sync`. Neural runs default to CUDA; use `neural.device=cpu` for small runs. Apple MPS does not support the required float64 ridge algebra.

```text
configs/                       Hydra defaults, tuning grids and checkpoints
scripts/
  embed_rxrx1.py                Images → Phenom crop embeddings
  train_batch_transfer_expression.py   Replogle fixed settings
  train_batch_transfer_rxrx1.py        RxRx1 fixed settings
  tune_bridges.py               Replogle nested tuning (shared or per-x)
  tune_bridges_rxrx1.py         RxRx1 nested tuning
src/genetic_perturbation_playground/
  data/                        Dataset loading and preprocessing
  model/batch_transfer.py       Linear models, bridge adapters and scaling
  model/proxy_unconfounded.py   Kernel bridge
  model/proxy_unconfounded_neural.py   Both neural bridges
  model/phenom.py               Frozen OpenPhenom encoder
  tuning.py                    Shared tuning logic
  utils/                       Kernels, metrics and plots
data/                          Downloaded data, embeddings and OpenPhenom weights
notebooks/                     Exploration and derivations
tests/                         Model, data and evaluation checks
hpc/run.sbatch                 Slurm GPU launcher
outputs/<date>/<name>_<time>/   Results, plots and .hydra/ configuration snapshots
```

The `train_proxy_confounded.py` and `train_proxy_unconfounded.py` scripts are separate synthetic experiments.

**Replogle:** the first run downloads and caches the data. `name` is required.

```bash
uv sync
uv run python scripts/train_batch_transfer_expression.py name=replogle
uv run python scripts/tune_bridges.py name=replogle_tuned
uv run python scripts/tune_bridges.py --config-name tune_bridges_perx name=replogle_perx
```

For linear models plus the kernel bridge on CPU, append `'bridges=[kernel]'` to a **training** command; `'bridges=[]'` runs only linear models. Tuning scripts always tune both bridges. A fixed per-$x$ Replogle run uses `neural.per_perturbation=true`.

**RxRx1:** download the images and metadata from the [dataset page](https://www.rxrx.ai/rxrx1). Arrange them as:

```text
data/rxrx1/rxrx1/metadata.csv
data/rxrx1/rxrx1/images/<experiment>/Plate<plate>/<well>_s<site>_w<channel>.png
```

The embedding script downloads OpenPhenom code and weights into `data/OpenPhenom/`. It embeds every metadata row by default. Generate embeddings before training; match `embeddings.grid` and `embeddings.standardise_sites` to the embedding run.

```bash
uv run python scripts/embed_rxrx1.py grid=4
uv run python scripts/train_batch_transfer_rxrx1.py name=huvec wandb_mode=disabled
uv run python scripts/tune_bridges_rxrx1.py name=all_types_tuned cell_type=null wandb_mode=disabled
```

Use `grid=2` when embedding and `embeddings.grid=2` when training to use 256-pixel crops. For site normalisation, use `standardise_sites=true` and `embeddings.standardise_sites=true` respectively. RxRx1 defaults to online W&B logging; examples disable it. Add `+neural.per_perturbation=true` to an RxRx1 command for the per-$x$ neural version.

To check the existing local smoke embeddings without a GPU:

```bash
uv run python scripts/train_batch_transfer_rxrx1.py name=smoke cell_type=null n_validation_batches=2 'bridges=[]' wandb_mode=disabled
```

For an embedding smoke run, `experiments='[HEPG2-01]'` limits the input. The output filename does not include this subset, so use a separate checkout/data directory to avoid replacing full embeddings.

**Slurm:** create the log directory, then pass a script and its Hydra overrides to the launcher. Defaults request one A100 GPU, 8 CPUs, 96 GB RAM and 12 hours; override resources for the cluster/job.

```bash
mkdir -p outputs
sbatch -J replogle hpc/run.sbatch scripts/train_batch_transfer_expression.py name=replogle
sbatch -J perx --time=2-00:00:00 hpc/run.sbatch scripts/tune_bridges.py --config-name tune_bridges_perx name=perx
```

**Results:** Replogle writes `scores_per_perturbation.csv`, `scores_per_batch.csv` and plots. Its tuning script also saves per-pair MSE in `scores_per_pair.csv`. RxRx1 writes per-pair `scores.csv` and overall `summary.csv`. Both tuning scripts save `inner_scores.csv` and `selected.csv`. Coverage plots are under `plots/by_training_batches/batch_<label>/`. Model weights are not saved by these entry points. Use each run's `.hydra/config.yaml` and logs to interpret archived results; older runs can differ in preprocessing, scaling and validation splits.
