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

The dataset loader downloads the [scPerturb H5AD](https://exampledata.scverse.org/pertpy/replogle_2022_k562_essential.h5ad) and applies these steps in order:

1. Remove cells with fewer than 100 total counts.
2. Remove genes detected in fewer than 5 cells.
3. Normalise each cell to 4,000 total counts, then apply `log1p`.
4. Keep single-perturbation cells (`nperts == 1`) and controls.
5. Save `data/replogle_k562_preprocessed.h5ad` and reuse it on later runs.

The current pipeline keeps all 8,563 retained genes. Older runs used 5,000 highly variable genes. The cache is not versioned: move an old cache aside before rerunning if it contains the older preprocessing.

## RxRx1

RxRx1 contains 125,510 microscopy sites, each with six 512 × 512, 8-bit channels stored as separate PNGs. There are two sites per well. These are images of cell populations; the pipeline does not segment or count individual cells. See the [dataset documentation](https://github.com/recursionpharma/rxrx-datasets/blob/trunk/rxrx1/README.md).

A **batch is one plate within an experiment** by default (`batch_unit=plate`), labelled for example `HUVEC-01|plate_1`. With `batch_unit=experiment`, all plates in an experiment form one batch, such as `HUVEC-01`. The choice controls proxy pools, response means, validation splits and training-batch counts in both training and tuning. Plate validation can train on other plates from the same experiment; experiment validation holds out all its plates together. All sites and crops from a plate stay together. Each experiment contains one cell type and four 384-well plates, with 308 usable wells per plate. There are **51 experiments**: 24 HUVEC, 11 HEPG2, 11 RPE and 5 U2OS. The design has **1,108 treatment siRNAs**, **30 positive-control siRNAs**, and untreated negative controls (`EMPTY`). Each treatment appears once per experiment; each positive control appears on every plate. Some intended treatments are missing or relabelled `EMPTY`. See the [RxRx1 paper](https://arxiv.org/html/2301.05768v1#S2).

A perturbation is the **(cell type, siRNA) tuple**. With `cell_type=null`, the code labels it `<cell_type>|<siRNA>`. The full design gives 4,432 treatment tuples and 120 positive-control tuples. With the default `cell_type=HUVEC`, cell type is fixed and labels contain only the siRNA. Selected proxies share the label `control`, but their distributions remain separate by batch.

Preprocessing in `scripts/embed_rxrx1.py` is:

1. **Crop:** split each site into a regular, non-overlapping `grid × grid`. The default `grid=4` gives 16 crops of 128 × 128 pixels per site. `grid=2` gives four 256 × 256 crops. Keep all crops, including empty or low-intensity ones.
2. **Standardise images:** divide pixel values by 255, then apply instance normalisation independently to each channel of each crop (approximately zero mean and unit variance, with numerical epsilon). With `standardise_sites=true`, normalise each full site before cropping instead.
3. **Embed:** use the frozen [OpenPhenom CA-MAE ViT-S/16 encoder](https://huggingface.co/recursionpharma/OpenPhenom). Keep all six channels and the crop's native resolution. Rebuild its fixed positional embeddings for the crop size. Drop the class token and average patch tokens.
4. **Save:** each crop becomes one **384-dimensional vector**. Store vectors and batch, cell type, siRNA, well, site and crop labels in H5AD. There is no PCA, embedding batch correction or fine-tuning of OpenPhenom.

For all 125,510 sites, the final matrix is **2,008,160 × 384** with `grid=4`, or **502,040 × 384** with `grid=2`. Its path is `data/rxrx1_phenom_<grid>x<grid>[_sitenorm].h5ad`.

The local checkout currently contains only a smoke subset: 88 sites from four experiments (HEPG2-01/02 and HUVEC-01/02), giving **1,408 × 384** embeddings. Older full runs used a removed low-intensity filter and recorded 497,208 crops at `grid=2`; those counts do not describe the current pipeline.

# Methods

Let $x$ be a perturbation, $e$ a batch and $g$ one of $G$ output features (genes or embedding coordinates). Let $\bar y_{xe}$ be the mean response, $n_{xe}$ its cell/crop count, and $w_{ej}$ a control sample. Linear models use the control mean $W_e$; bridges use the full control distribution.

**Input standardisation.** By default (`standardise_w=true`), every continuous proxy coordinate uses $\tilde w_g=(w_g-a_g)/s_g$. The mean $a_g$ and population standard deviation $s_g$ are fitted to pooled training controls only, and reused for validation and test controls. Constant coordinates use $s_g=1$. Linear control means use the same transformation. Categorical/one-hot perturbation inputs stay unchanged. Targets and predictions remain in log-expression or Phenom units. Image normalisation above is a separate step.

**Count weighting.** Linear ridge always uses cell/crop counts. For bridges, $c_{xe}=n_{xe}$ by default; `weight_by_cells=false` sets $c_{xe}=1$ for observed pairs. **If perturbation $x$ is absent from batch $e$, its weight is zero.** Missing responses are omitted, not treated as observed zeros. Let $N=\sum_{xe}c_{xe}$. Ridge penalties use $\lambda N$, where $N$ covers all perturbations in the current training split. Each batch's control embedding is an unweighted mean over its proxy samples.

## 1. Linear

Each feature has a perturbation intercept $\gamma_{xg}$ and a batch slope $b_g$ shared across perturbations:

$$
\hat y_{xeg}=\gamma_{xg}+b_g W_{eg},\qquad
\min_{\gamma,b}\sum_{x,e}n_{xe}\sum_g(\bar y_{xeg}-\gamma_{xg}-b_gW_{eg})^2
+N\lambda\sum_g(s_gb_g)^2.
$$

The intercepts $\gamma$ have no penalty. Here, $s_g$ is the training-control standard deviation when inputs are standardised, and 1 otherwise. Thus the penalty applies to the fitted slope in the chosen control units. `LinearTransfer(reg=...)` from `model/linear_transfer.py` sets $\lambda$; training and tuning scripts use `linear.reg`, which defaults to `1e-3`. Set it to `0` for ordinary least squares. Fixed slopes (`b=0` and `b=1`) ignore this penalty.

Define the training means $\bar y_{xg}=\sum_e n_{xe}\bar y_{xeg}/\sum_e n_{xe}$ and $\bar W_{xg}=\sum_e n_{xe}W_{eg}/\sum_e n_{xe}$. The three settings predict a new batch $e^*$ as follows:

| Setting | Prediction |
| --- | --- |
| Pooled (`b=0`) | $\hat y_{xe^*g}=\bar y_{xg}$ |
| Shift (`b=1`) | $\hat y_{xe^*g}=\bar y_{xg}-\bar W_{xg}+W_{e^*g}$ |
| Transfer (fit $b$) | $\hat y_{xe^*g}=\bar y_{xg}+\hat b_g(W_{e^*g}-\bar W_{xg})$ |

The fitted ridge slope is:

$$
\hat b_g=
\frac{\sum_{x,e}n_{xe}(W_{eg}-\bar W_{xg})(\bar y_{xeg}-\bar y_{xg})}
{\sum_{x,e}n_{xe}(W_{eg}-\bar W_{xg})^2+N\lambda s_g^2}.
$$

These equations use original units. The implementation standardises inputs and converts fixed slopes accordingly, so `b=1` still means a unit shift in original units. Each output uses only its matching control coordinate.

## 2. Kernel bridge

The perturbation feature map is the delta/one-hot basis: $\phi(x)=\delta_x$, so $k_x(x,z)=\mathbf{1}[x=z]$. For two standardised control vectors $u$ and $v$, use an RBF kernel:

$$
k_w(u,v)=\exp\left(-\frac{\Vert u-v\Vert^{2}}{2\ell^{2}}\right),\qquad
\mu_e=\frac{1}{m_e}\sum_{j=1}^{m_e}\psi(\tilde{w}_{ej}).
$$

Here $m_e$ is the number of proxy samples and $\psi$ is the RBF feature map. The similarity between batches $e$ and $f$ is $M_{ef}=\langle\mu_e,\mu_f\rangle$, computed by averaging kernel values over all pairs of controls. The lengthscale is the median pairwise distance of at most 2,000 pooled training controls, multiplied by `lengthscale_scale`.

The bridge predicts $\hat{y}_{xeg}=\langle h_g,\phi(x)\otimes\mu_e\rangle$. The delta map gives a separate ridge fit per perturbation, with shared batch similarities. On the batches where $x$ occurs, solve:

$$
D_x=\mathrm{diag}(1/c_{xe}),\qquad
(M_x+\lambda N D_x)\alpha_x=\bar{Y}_x.
$$

For a new batch $t$, predict:

$$
\hat{y}_{xt}=M_{t,x}\alpha_x.
$$

$\bar{Y}_x$ contains the observed batch means, and $M_x$ contains similarities between those training batches. $M_{t,x}$ compares new-batch controls with those training batches. There is no fitted response sharing across perturbations.

## 3. Neural bridge

The default neural model is **per-x**: one control network maps each sample to $q$ features, and its batch mean is $z_e=m_e^{-1}\sum_j\psi(\tilde w_{ej})$. Each perturbation has its own response matrix:

$$
\hat y_{xe}=z_e^\top A_x,\qquad
\min_{\{A_x\}}\sum_{x,e}c_{xe}\|\bar y_{xe}-z_e^\top A_x\|^2
+\lambda N\sum_x\|A_x\|_F^2.
$$

Only the control network is shared across perturbations. `neural=per_x` selects this model; it is also the default. The learned-perturbation marginal neural model has been removed. The kernel marginal embedding bridge is still available.

The control networks use full-batch AdamW, float32 SwiGLU layers with residual connections, and float64 ridge algebra. `n_features_w` sets the embedding size, `width_w` the hidden width, and `depth` the hidden-layer count. Each training step uses a fresh environment split when `cross_fit_folds >= 2`; `cross_fit_folds=0` uses in-sample MSE. Final coefficients use every training environment. Training and checkpoint callbacks are shared by all three variants.

### Shared 2NN and hybrid

`NeuralSharedEmbeddingBridge` learns two independent control networks. Write their batch means as $s_e$ and $d_e$, each with `n_features_w` coordinates. The shared 2NN prediction is

$$
\hat y_{xeg}=a_{xg}+s_e^\top b_g+d_e^\top c_{xg}.
$$

The intercept $a_x$ and deviation coefficients $c_x$ belong to each perturbation; $b$ is shared across perturbations. For fixed networks, all coefficients are fitted jointly with the penalty

$$
N\left[\lambda\left(\sum_x\|a_x\|^2+\|b\|_F^2\right)
+\lambda_c\sum_x\|c_x\|_F^2\right].
$$

`reg` sets $\lambda$; `deviation_reg` sets $\lambda_c$ and defaults to `reg`. Both must be finite and positive. The intercept is penalised too. Increasing `deviation_reg` shrinks perturbation-specific deviations toward the shared effect.

`NeuralHybridEmbeddingBridge` adds $\beta_g W_{eg}$ to this prediction and $N\lambda_{\rm linear}\|\beta\|^2$ to the penalty. The linear slope, intercept, shared response and deviation are fitted **jointly**. Each raw-control coordinate must correspond to the same target coordinate. `linear_reg` is nonnegative; `float("inf")` disables this term and reproduces shared 2NN. If controls are standardised, the linear term uses the standardised control mean.

Both models retain count weighting, missing-pair handling, float64 ridge algebra, float32 networks, cross-fitting and checkpoint callbacks. `width_w`, `depth` and `n_features_w` size each network. The training loss is prediction MSE, without the coefficient penalty. `Standardised` centres training targets and adds their mean back at prediction, so coefficients in that wrapper describe centred responses.

Use either variant through the existing AnnData adapter:

```python
from genetic_perturbation_playground.model.batch_transfer import NeuralBridgeTransfer
from genetic_perturbation_playground.utils.standardisation import Standardised

model = Standardised(NeuralBridgeTransfer(
    variant="hybrid",  # "shared_2nn" omits the raw-linear term
    reg=1e-4,
    deviation_reg=1e-3,
    linear_reg=1e-3,  # hybrid only
    n_features_w=16,
    width_w=32,
    cross_fit_folds=5,
    n_steps=400,
))
model.fit(train)
predictions = model.predict(perturbations, test)
```

The training scripts accept `neural=shared_2nn` or `neural=hybrid`. Set variant-specific penalties with `neural.deviation_reg=1e-3` and `neural.linear_reg=1e-3` (hybrid only). The default is `per_x`. There is no `marginal` neural variant.

Model settings live in the dataclasses in `src/genetic_perturbation_playground/config.py`: `LinearConfig`, `KernelConfig`, `PerXConfig`, `Shared2NNConfig` and `HybridConfig`. All neural models default to one hidden layer per MLP; `neural.depth` excludes the output layer. The existing model constructors still accept keyword arguments. To reuse a named set of settings in Python:

```python
from genetic_perturbation_playground.config import HybridConfig
from genetic_perturbation_playground.model.batch_transfer import build_neural

settings = HybridConfig(width_w=32, n_features_w=16, n_steps=400)
model = build_neural(settings, standardise_w=True)
model.fit(train)
```

Hydra applies the dataclass defaults, then `configs/transfer_defaults.yaml`, the dataset/tuning preset, and command-line overrides. Tuning applies each sampled trial last, without changing the experiment config. Direct Python defaults use CPU; the shared transfer preset selects CUDA. Replogle uses width 64 and RxRx1 uses width 128. Tuning grids stay in YAML.

Use `linear.reg` for the standalone linear model, `kernel.reg` for the kernel bridge and `neural.reg` for neural ridge. The hybrid's `neural.linear_reg` is a separate penalty. Setting `neural.deviation_reg=null` makes it follow that trial's `neural.reg`. Model fields are checked against the selected config group, including fields added with `+`.

Older commands need these changes:

| Previous option | Current option |
|---|---|
| `neural.variant=hybrid` | `neural=hybrid` |
| `neural.variant=shared_2nn` | `neural=shared_2nn` |
| `linear_reg=0` | `linear.reg=0` |
| `bridge_reg=1e-5` | `kernel.reg=1e-5 neural.reg=1e-5` |
| `+neural.linear_reg=0` | `neural=hybrid neural.linear_reg=0` |
| `+neural.deviation_reg=1e-3` | `neural=shared_2nn neural.deviation_reg=1e-3` (or select hybrid) |

For programmatic Hydra composition, call `register_configs()` from the config module before `hydra.compose`, then use `resolve_model_config(cfg, "neural")` to obtain the selected dataclass. Use Hydra composition to load transfer presets; a raw YAML load does not apply their inherited defaults.

`model/batch_transfer.py` contains the kernel and neural AnnData adapters. `data/transfer.py` prepares their counts, means and batch mixtures. The linear model is `LinearTransfer` in `model/linear_transfer.py`, and the scaling wrapper is `Standardised` in `utils/standardisation.py`.

`utils/neural.py` contains the reusable network layers, network builder, tensor conversion and checkpoint helper. `data/neural.py` defines one `GroupedRidgeData` record shared by the neural models and ridge solver; it holds counts, target sums and optional control tensors without copying them.

Implementation is in `model/proxy_unconfounded_neural.py`. `utils/ridge.py` contains two independent solvers: `LinearRidgeSolver` fits NumPy control slopes with a ridge penalty and unpenalised category intercepts; `NeuralRidgeSolver` fits the coefficients of all three neural bridges with PyTorch gradients. They have no common base class. Per-x retains batched category solves and target-Gram losses. For shared 2NN and hybrid, the neural solver eliminates category coefficients before a small shared solve, then solves the hybrid's extra slope per coordinate. Shared-2NN training uses target Grams and batches all folds' target projections; hybrid training checkpoints folds to limit retained autograd state. Final shared/hybrid coefficients are reconstructed in `gene_block_size` blocks and exposed as `intercept_`, `shared_coef_`, `deviation_coef_` and `linear_coef_` (zero for shared 2NN).

`scripts/benchmark_ridge_solver.py --baseline /path/to/previous_neural_module.py --output outputs/ridge_benchmark.json` compares a saved pre-refactor module against the neural solver. It checks losses, gradients and coefficients, then records CPU training/reconstruction timings and allocation traffic on two synthetic sizes. The linear comparison sets `reg=0` to match the previous least-squares formula.

## Virtual experiments

`predict_environment_average(perturbations, reference)` predicts fitted perturbations over a common distribution of environments. `reference` contains only control rows, with `obs["perturbation"] == "control"` and environment names in `obs["batch"]`. Its feature names and order must match training. By default, each environment has equal weight, regardless of its sample count. The method works on linear, kernel and all three neural transfer models, including `Standardised` wrappers.

`virtual_experiments` returns labeled predictions and can also retain observed treatment means. Let $\widehat y_{xeg}$ be the bridge prediction for perturbation $x$ in environment $e$, averaged over that environment's controls. The two estimates are

$$
\widehat\mu^{(1)}_g(x)=\sum_e p(e)\widehat y_{xeg},\qquad
\widehat\mu^{(2)}_g(x)=\widehat\mu^{(1)}_g(x)+\sum_{e\in E_x}p(e)(\bar y_{xeg}-\widehat y_{xeg}).
$$

Use an already-fitted model and control rows from the reference environments:

```python
from genetic_perturbation_playground.model.virtual_experiments import virtual_experiments

reference = reference_data[reference_data.obs["perturbation"].eq("control")]
result = virtual_experiments(model, reference)  # All fitted perturbations; uniform environments.
bridge_means = result.X
```

For two reference environments named `e0` and `e1`, explicit probabilities and separately supplied responses give both estimates:

```python
result = virtual_experiments(
    model,
    reference,
    environment_weights={"e0": 0.25, "e1": 0.75},
    observed=available_responses,
)
bridge_means = result.X
filled_means = result.layers["observed_fill"]
coverage = result.obs["observed_weight"]
weights = result.uns["environment_weights"]
```

Rows are labeled by perturbation; columns retain the reference feature annotations. `observed` may contain response rows without controls. Its selected pair means replace predictions only for those pairs; missing pairs are absent rows, and zero responses count as observations. Coverage is the sum of environment probabilities with an observed response, not a fraction of cells. Neither function fits a model or changes its inputs. `Standardised` reuses training scales and restores the target mean, so its results and supplied observations are in original target units.

For RxRx1, use **HUVEC only** when fitting and choosing reference/observed rows. Environment labels are supplied by the caller. Equal weights define a reference population; a causal interpretation requires the bridge assumptions and appropriate observed intervention means. Estimator 2 is not automatically doubly robust.

For evaluation, exclude held-out responses from both fitting and `observed`, and compare predictions under the same reference distribution. Internal neural cross-fitting does not replace an outer evaluation split. Neural inference averages network outputs before applying coefficients; it does not apply the network to averaged controls. Corrections are accumulated one environment at a time, without constructing a perturbation-by-environment-by-output array.

### RxRx1 sparse-response recovery

`scripts/evaluate_rxrx1_sparse_recovery.py` tests whether virtual averaging recovers the full observed mean better than the sparse observed average. For each seed, independently permute each treatment's originally observed batches and retain the first `k=2,3,4`; these retained sets are nested across k. All selected proxy controls remain available, including controls in batches with no retained treatment responses. Defaults are HUVEC, plate batches and positive controls. Positive controls never become response targets in this experiment, even with negative-only proxies.

The full target and retained baseline average crop means **equally across batches**. Each virtual estimate averages predictions over that perturbation's own original reference batches, including retained ones. It uses the pure bridge prediction, without observed-value replacement. Score each estimate by mean squared error over embedding coordinates against the full target. Fitting keeps the existing crop-count weights and retained-response target centering.

Tuning is repeated independently for every `(seed,k)` pair. Each of k inner folds holds out one retained response batch per perturbation, leaving k-1 batches for fitting and retaining all controls. Select settings by MSE averaged equally over perturbations, then folds, and refit on all k retained batches. Removed responses and full-data means never enter tuning. `neural.cross_fit_folds=0` in this preset because k=2 inner fits have only one response batch per perturbation. The neural variant can be selected with `neural=per_x|shared_2nn|hybrid`.

Sample **8 configurations per bridge** without replacement from the existing grids once, reusing that draw across seeds and k. `n_configs` changes the subsample size; neural scoring checkpoints remain unchanged. With 10 seeds and the default grids, this schedules 720 kernel tuning fits, 720 neural tuning runs and 30 final refits per bridge. The runner prints the workload before fitting; kernel similarities are currently recomputed by the existing model implementation.

```bash
uv run python scripts/evaluate_rxrx1_sparse_recovery.py name=huvec_sparse n_seeds=10 seed=0 n_configs=8
# Repeat the identical command to resume. Increase n_seeds to append new seeds.
```

Results live in the stable folder `outputs/rxrx1_sparse_recovery/huvec_sparse/`; override `output_dir` to choose another location. Invocation logs have timestamped subfolders. The runner saves each completed candidate/fold trial, partial neural scores at every scoring checkpoint, each completed seed/k result, and each completed seed. Resume skips completed trials and stages; an interrupted neural trial restarts from the beginning. An interrupted final refit reuses completed tuning. There is no optimizer-state checkpoint. A lock prevents concurrent writers, and content hashes reject changed inputs, relevant source dependencies or scientific settings. `resume=false` rejects existing results. Reducing `n_seeds` preserves earlier completed results in the combined tables.

The hybrid's documented `linear_reg=inf` setting is supported in grids and configuration. Saved settings encode it as `{"__float__":"+inf"}`; use `genetic_perturbation_playground.experiments.checkpoints.decode_settings` after reading JSON to restore it. Scores must always be finite. The experiment seed controls both retention and neural initialization; independent `neural.seed` values and `seed` entries in the active neural grid are rejected.

The output folder contains:

- `manifest.json`, `reference_pairs.csv`, `eligibility.csv`: resolved scientific identity, sampled candidates, original reference memberships and coverage exclusions.
- `seeds/seed_<seed>/retained_order.csv` and `effective_config.json`: the repeatable batch draw and effective settings.
- `seeds/seed_<seed>/k_<k>/trials/`: one committed JSON per candidate/fold; `.partial.json` files are progress only.
- Each seed/k folder's `inner_scores.csv`, `selected.json`, `estimates.npz`, `scores.csv` and `summary.csv`: validation scores, winners, labeled full/retained/virtual mean vectors, per-perturbation MSE and paired summaries. NPZ arrays load with `allow_pickle=False`.
- Root `scores.csv` and `summary.csv`: rebuilt from validated completed stages on every invocation, including an otherwise fully skipped resume.

Score rows contain `seed`, `k`, `perturbation`, `n_reference_batches`, `n_retained_batches`, `n_removed_batches`, `model`, `metric`, `value`. Models are `retained_mean`, `kernel_virtual`, `neural_virtual`; the metric is `mse`. Summaries report baseline and virtual MSE, `improvement = baseline_mse - virtual_mse` (positive is better), strict improvement fraction, cohort sizes, seed counts and sample SD across seed means. They report both all eligible perturbations and `removed_only`, which excludes exact-k references. Empty cohorts have zero counts and missing metrics; a one-seed SD is missing.

Perturbations with fewer than k original batches are explicitly excluded for that k. Exact-k references remain as diagnostics with zero baseline error. If any requested k has no eligible perturbations, the runner saves `reference_pairs.csv` and `eligibility.csv`, reports all empty cohorts and stops before fitting. The small local fixture has only two HUVEC batches per treatment and negative controls; the default experiment requires the full embeddings. Tests use synthetic data with sufficient coverage and do not launch the full experiment.

# Evaluation

Select distinct validation batches using `seed` (default 0). Hold out each selected batch separately and train on **all other batches**, including those held out in other runs. Defaults are 10 outer evaluations for Replogle and 5 for RxRx1. Score only perturbations present in both the training data and the held-out batch. This evaluates new batches for known perturbations.

**Replogle controls and effects.** Randomly split every batch's controls into two near-equal halves, including training batches. One half supplies proxies and input-scaling statistics. The other half is reserved for evaluation baselines. For each held-out perturbation, subtract that second half's mean $W_e^{\mathrm{eval}}$ from both observed and predicted expression:

$$
\Delta_{xe}=\bar y_{xe}-W_e^{\mathrm{eval}},\qquad
\hat\Delta_{xe}=\hat y_{xe}-W_e^{\mathrm{eval}}.
$$

**All Replogle metrics compare differential-expression vectors** $\hat\Delta_{xe}$ and $\Delta_{xe}$. These effects are differences in mean log-normalised expression; there is no significance test.

| Dataset | Metrics for each (perturbation, held-out batch) |
| --- | --- |
| Replogle | On differential expression: MSE averaged over all genes; Pearson correlation across all genes; `pearson_top100`, across the 100 genes with largest absolute observed differential expression (`top_k` is configurable). MSE also includes a **no effect** baseline, $\hat\Delta=0$. |
| RxRx1 | MSE averaged over the 384 coordinates of predicted and observed mean crop embeddings. No control subtraction. |

Subtracting the same control vector from prediction and truth leaves MSE unchanged, so model MSE also equals mean-expression MSE. Pearson correlations and top-gene selection depend on the differential-expression vectors.

RxRx1 selects proxies with `proxy_controls=negative|positive|both` in both training and tuning, with no half split. The selection applies to training, inner-validation and held-out batches:

- `proxy_controls=negative`: only negative-control crops are proxies. Positive controls remain training perturbations.
- `proxy_controls=positive` (default): only positive-control crops are proxies. Negative-control crops are excluded from the model data.
- `proxy_controls=both`: negative- and positive-control crops are pooled as proxies.

Only treatment wells are evaluated; positive controls are never scored. Each response mean averages all crops for that perturbation in that batch. Batches without any selected proxy controls are rejected. Both scripts default to `batch_unit=plate proxy_controls=positive`; use `batch_unit=experiment proxy_controls=negative` for experiment batches with negative-only proxies. Existing embeddings already contain plate metadata, so switching batch units does not require embedding again.

Every validation batch must contain at least one treatment also present in its training split. Outer and inner splits that violate this are rejected before fitting that split, with the affected batch IDs; use more training batches or adjust the validation seed/folds.

**Aggregation:** first compute one metric across features for each observed perturbation–batch pair. Then take arithmetic means, without cell/crop weighting:

- **Per batch:** average over its scored perturbations. The overall batch-based result averages these batch scores equally. RxRx1 writes this result to `summary.csv`.
- **Per perturbation (Replogle):** average over selected held-out batches containing that perturbation. The per-perturbation report then averages these scores equally. This can differ from the batch-based result because coverage differs.

Cell counts in score tables describe coverage; they do not weight evaluation averages. `n_training_batches` counts training batches containing each perturbation and is used in coverage plots. Undefined Pearson values are NaN and are skipped by pandas averages.

**Tuning:** nested cross-validation keeps the outer batch out of tuning. Split the remaining batches into three inner folds. Randomly sample bridge settings from the YAML grids and score neural training checkpoints. Select the lowest mean inner MSE: average over perturbations within each batch, then batches within an inner fold, then inner folds equally. Refit on all outer-training batches. The same candidate draw is reused across outer evaluations. When pooling RxRx1 cell types, inner folds distribute batches by cell type. These inner folds are separate from neural cross-fitting.

# Paths and how to run the experiments

Run commands from the repository root. Use Python 3.12+ and install the locked environment with `uv sync`. Neural runs default to CUDA; use `neural.device=cpu` for small runs. Apple MPS does not support the required float64 ridge algebra. Set `MPLBACKEND=Agg` for headless plotting.

```text
configs/                       Hydra defaults, tuning grids and checkpoints
scripts/
  embed_rxrx1.py                Images → Phenom crop embeddings
  train_replogle_transfer.py    Replogle fixed settings
  train_rxrx1_transfer.py       RxRx1 fixed settings
  tune_replogle_transfer.py     Replogle nested tuning
  tune_rxrx1_transfer.py        RxRx1 nested tuning
src/genetic_perturbation_playground/
  config.py                    Typed model settings and Hydra registration
  data/                        Dataset loading and preprocessing
  model/batch_transfer.py       Kernel and neural bridge adapters
  model/linear_transfer.py      Linear transfer
  model/proxy_unconfounded.py   Kernel bridge
  model/proxy_unconfounded_neural.py   Per-x, shared 2NN and hybrid bridges
  model/phenom.py               Frozen OpenPhenom encoder
  tuning.py                    Shared tuning logic
  utils/                       Ridge solvers, networks, scaling, kernels, metrics and plots
data/                          Downloaded data, embeddings and OpenPhenom weights
notebooks/                     Exploration and derivations
tests/                         Model, data and evaluation checks
hpc/run.sbatch                 Slurm GPU launcher
hpc/run_beaker.qsub            Beaker Grid Engine GPU launcher
outputs/<date>/<name>_<time>/   Results, plots and .hydra/ configuration snapshots
```

The `train_proxy_confounded.py` and `train_proxy_unconfounded.py` scripts are separate synthetic experiments. The unconfounded continuous-X experiment now runs only the kernel bridge, since its neural marginal model has been removed.

**Replogle:** the first run downloads and caches the data. `name` is required.

```bash
uv sync
uv run python scripts/train_replogle_transfer.py name=replogle
uv run python scripts/train_replogle_transfer.py name=replogle_shared neural=shared_2nn
uv run python scripts/tune_replogle_transfer.py name=replogle_tuned
uv run python scripts/tune_replogle_transfer.py name=replogle_shared_tuned neural=shared_2nn
```

For linear models plus the kernel bridge on CPU, append `'bridges=[kernel]'` to a **training** command; `'bridges=[]'` runs only linear models. Tuning scripts always tune a kernel bridge and one neural variant: `neural=per_x` (default), `neural=shared_2nn` or `neural=hybrid`.

Each tuning config samples 32 neural configurations for the selected model and evaluates all kernel configurations (20 for Replogle, 30 for RxRx1). Networks use one hidden layer. `neural_grid` holds common ranges; `neural_variant_grids` adds the separate deviation penalty for shared 2NN and the deviation and linear penalties for hybrid. Explicit `neural_grid` settings override the variant ranges. Checkpoints include initialization and run through 400 steps; every sampled candidate trains to 400, with the best checkpoint chosen on inner validation.

**RxRx1:** download the images and metadata from the [dataset page](https://www.rxrx.ai/rxrx1). Arrange them as:

```text
data/rxrx1/rxrx1/metadata.csv
data/rxrx1/rxrx1/images/<experiment>/Plate<plate>/<well>_s<site>_w<channel>.png
```

The embedding script downloads OpenPhenom code and weights into `data/OpenPhenom/`. It embeds every metadata row by default. Generate embeddings before training; match `embeddings.grid` and `embeddings.standardise_sites` to the embedding run.

```bash
uv run python scripts/embed_rxrx1.py grid=4
uv run python scripts/train_rxrx1_transfer.py name=huvec wandb_mode=disabled
uv run python scripts/train_rxrx1_transfer.py name=huvec_hybrid neural=hybrid wandb_mode=disabled
uv run python scripts/train_rxrx1_transfer.py name=huvec_experiment batch_unit=experiment proxy_controls=negative wandb_mode=disabled
uv run python scripts/tune_rxrx1_transfer.py name=all_types_tuned cell_type=null wandb_mode=disabled
```

Use `grid=2` when embedding and `embeddings.grid=2` when training to use 256-pixel crops. For site normalisation, use `standardise_sites=true` and `embeddings.standardise_sites=true` respectively. RxRx1 defaults to online W&B logging; examples disable it. RxRx1 also defaults to `neural=per_x`.

To check the existing local smoke embeddings without a GPU:

```bash
uv run python scripts/train_rxrx1_transfer.py name=smoke cell_type=null n_validation_batches=2 'bridges=[]' wandb_mode=disabled
```

For an embedding smoke run, `experiments='[HEPG2-01]'` limits the input. The output filename does not include this subset, so use a separate checkout/data directory to avoid replacing full embeddings.

**Slurm:** create the log directory, then pass a script and its Hydra overrides to the launcher. Defaults request one A100 GPU, 8 CPUs, 96 GB RAM and 12 hours; override resources for the cluster/job.

```bash
mkdir -p outputs
sbatch -J replogle hpc/run.sbatch scripts/train_replogle_transfer.py name=replogle
sbatch -J replogle_tune --time=2-00:00:00 hpc/run.sbatch scripts/tune_replogle_transfer.py name=replogle_tuned
```

**Beaker:** use Grid Engine's `qsub`, not `sbatch`. The launcher requests one 80 GB A100 GPU, 96 GB host memory and 48 hours, following `/share/apps/examples/cluster_induction/one_gpu_test.sh`. GPU jobs use `tmem`; the CPU-job `h_vmem` limit is not needed. Override resources with `qsub -l ...`.

The launcher uses Beaker's installed `/share/apps/singularity_pytorch/pytorch_25.11-py3.sif` container, which has Python 3.12 and `uv`. It installs the locked project dependencies in `.venv-beaker`, separate from any host environment, and preserves the scheduler's `CUDA_VISIBLE_DEVICES` selection. This avoids the CentOS 7 host's older C library. The repository and datasets must already be available on Beaker's shared storage. To prepare dependencies before occupying a GPU, run from the repo root:

```bash
export BEAKER_IMAGE=/share/apps/singularity_pytorch/pytorch_25.11-py3.sif
SINGULARITYENV_UV_PROJECT_ENVIRONMENT="$PWD/.venv-beaker" \
  singularity exec --cleanenv --bind "$PWD:$PWD" --pwd "$PWD" "$BEAKER_IMAGE" \
  uv sync --frozen --no-dev
mkdir -p outputs
qsub -N replogle_shared hpc/run_beaker.qsub scripts/tune_replogle_transfer.py \
  name=replogle_shared neural=shared_2nn
qsub -N huvec_hybrid hpc/run_beaker.qsub scripts/tune_rxrx1_transfer.py \
  name=huvec_hybrid neural=hybrid wandb_mode=offline
```

Each submission runs one neural variant; change `neural` to run the other. Use `qsub -v BEAKER_IMAGE=/absolute/path/to/image.sif ...` for another image containing Python 3.12+ and `uv`. If `data/` points outside the bound repository and home directory, also bind its target with `SINGULARITY_BINDPATH`. `qsub -verify` prints the submission settings without queuing a job.

**Results:** Replogle writes `scores_per_perturbation.csv`, `scores_per_batch.csv` and plots. Its tuning script also saves per-pair MSE in `scores_per_pair.csv`. RxRx1 writes per-pair `scores.csv` and overall `summary.csv`. Both tuning scripts save `inner_scores.csv` and `selected.csv`. Coverage plots are under `plots/by_training_batches/batch_<label>/`. Model weights are not saved by these entry points. Use each run's `.hydra/config.yaml` and logs to interpret archived results; older runs can differ in preprocessing, scaling and validation splits.
