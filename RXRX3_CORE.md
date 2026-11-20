# RxRx3-core: reproducing the benchmark and the per-plate control shift (public data only)

Everything here runs on public artefacts only:

- the Hugging Face dataset [`recursionpharma/rxrx3-core`](https://huggingface.co/datasets/recursionpharma/rxrx3-core)
  (images, metadata and the four released embedding sets);
- the Hugging Face model [`recursionpharma/OpenPhenom`](https://huggingface.co/recursionpharma/OpenPhenom);
- the GitHub repo [`recursionpharma/EFAAR_benchmarking`](https://github.com/recursionpharma/EFAAR_benchmarking)
  (alignments, scorer and relationship annotations), commit `2935f21`.

The reference paper is arXiv 2503.20158 (Table 6). An earlier version of that paper reported
roughly twice the recall shown here (e.g. CORUM .654 for OpenPhenom), because it did not filter out
compound wells. The current version's Table 6 matches what the official code gives (Section 4.1).

## 1. Summary

1. **The official pipeline reproduces Table 6.** Running the official notebook's steps on the
   released embeddings gives the paper's RxRx3-core numbers to within 0.007 for CellProfiler,
   Phenom-1 and Phenom-2 on all four databases. OpenPhenom comes out 0.012–0.036 above the paper
   (Section 4.1).
2. **Our own OpenPhenom embedding code reproduces the released embeddings.** Re-embedding the
   public images gives a median per-well cosine of 0.987 to the released vectors, and the same
   recall (CORUM .322 vs .326).
3. **A per-plate control shift raises recall substantially.** After the official alignment,
   subtract part of each plate's control-well mean from every well on that plate. On CORUM, HuMAP
   and StringDB this lifts recall on every set, alignment and cutoff when half the shift is used:
   96 of 96 cells, with a mean gain of +0.057 recall after PCA-CS and +0.045 after TVN.
   - OpenPhenom (CORUM .326 → .428 at the 95th percentile, PCA-CS) moves to about Phenom-1 level
     (.398 without the shift).
   - Gains are relatively larger at the strict cutoffs: OpenPhenom CORUM at the 99.9th roughly
     doubles (.034 → .063).
4. **The full shift over-corrects.** It hurts Phenom-2 (CORUM .480 → .402 at the 95th) and
   collapses CellProfiler under EFAAR (.385 → .152). A bias–variance argument (Section 3.4) explains
   why a partial shift wins.
5. **The shift weight can be chosen without the benchmark.** An empirical-Bayes weight per plate,
   α_p = τ²/(τ² + v_p), estimated from the control wells alone:
   - beats no shift in 93 of 96 CORUM / HuMAP / StringDB cells (all 48 under PCA-CS);
   - is within 0.02 of the hand-picked half shift on average for every set;
   - fixes the CellProfiler EFAAR collapse (CORUM .419, the best map for that set).
6. **Reactome is not helped** by any shift (all within about ±0.02 of no shift, and the full shift
   lowers it). The gains are in complex and physical-interaction databases.

## 2. Setup

From the repo root:

```bash
uv sync
```

The official benchmark code is cloned, not installed. `scripts/rxrx3_core_repro.py` adds it to
`sys.path` and stubs its unused `geomloss` / `scvi` imports. Its CORUM, HuMAP, Reactome, SIGNOR and
StringDB annotations ship inside the repo.

```bash
git clone https://github.com/recursionpharma/EFAAR_benchmarking outputs/scratch/EFAAR_benchmarking
```

```bash
git -C outputs/scratch/EFAAR_benchmarking checkout 2935f21f84898fc26e2a9a40bdaee685c8e63fb6
```

The data is about 22 GB (17 GB of it images):

- `metadata_rxrx3_core.csv`;
- the four released embedding parquets: CellProfiler, OpenPhenom, Phenom-1 (`MAE-L_8_RPI-93M`)
  and Phenom-2 (`MAE-G_8_PP-16M_trimmed`);
- 35 image shards, `data/train-000NN-of-00035.parquet`. Each row is one 512×512 channel image
  (JPEG 2000, native resolution centre crop), keyed `<experiment>/Plate<p>/<address>_s<site>_<channel>`.

```bash
uv run hf download recursionpharma/rxrx3-core --repo-type dataset --local-dir outputs/scratch/rxrx3_core
```

`model/phenom.py:load_phenom` downloads `recursionpharma/OpenPhenom` on first use.

### The two self-contained scripts

`scripts/rxrx3_core_minimal_repro.py` and `scripts/rxrx3_core_proxy_minimal_repro.py` don't need
this repo, the clone or the download above. They fetch what they need from Hugging Face
themselves. In a fresh environment:

```bash
pip install "efaar_benchmarking @ git+https://github.com/recursionpharma/EFAAR_benchmarking@2935f21" huggingface_hub pyarrow matplotlib
```

```bash
python rxrx3_core_proxy_minimal_repro.py
```

## 3. Methods

Notation: x is a well's feature vector (wells with several rows are averaged first). Each well
belongs to experiment b and plate p (`experiment_name/plate`). "Controls" means the
`EMPTY_control` wells unless stated otherwise.

### 3.1 Scoring

For each map:

1. Align the well features (Section 3.2) and, optionally, shift them (Section 3.3).
2. Average the CRISPR wells of each non-control treatment into a gene vector. Guides map to genes
   via `treatment.split("_")[0]`. Only genes are scored (the paper lists 736).
3. Score with the official `known_relationship_benchmark`. For each database, recall is the share
   of known pairs whose cosine falls in either tail of the distribution of all gene-pair cosines:
   below the (1 − q) quantile or above the q quantile. The scripts report q = 0.95, 0.99, 0.995 and
   0.999 ("both tails"). The benchmark draws the null by random sampling, so reruns move by up to
   about 0.003.

### 3.2 Alignments

All three are per-batch affine maps x ↦ (x − a_b) B_b, fit on control wells only.

- **PCA-CS** (`efaar.pca_centerscale_on_controls`, batch = experiment). Global PCA rotation fit
  on all controls (all components kept), then per-experiment standardisation of each component
  on that experiment's controls. It aligns each batch's control mean and per-axis variance, in one
  global basis.
- **TVN** (`efaar.tvn_on_controls`, batch = experiment).
  1. Standardise each feature on all controls (no batch split).
  2. PCA rotation fit on the controls.
  3. Per-experiment standardisation, as in PCA-CS.
  4. Per-experiment CORAL: z ↦ z (C_b + εI)^(−1/2) (C + εI)^(1/2), with ε = 0.5, C_b the batch's
     control covariance and C the pooled one. Directions with control variance well below ε are
     barely rescaled.

  TVN also aligns correlations, not just variances.
- **EFAAR** (`rxrx3_core_proxy.efaar`, batch = plate). The TVN recipe with two changes: the batch
  is the plate (about 1,744 of them), and step 1 standardises per plate. Afterwards every plate's
  EMPTY mean is exactly 0.

### 3.3 The control shift

Let C_p be plate p's control wells: its `EMPTY_control` wells together with its
`well_type_label == "CRISPR Gene Positive Controls"` wells (about 16 + 12 on a gene plate; the
positive controls are guides for genes such as PLK1, MTOR and SRC). On aligned features y:

- m_p is the mean of y over C_p;
- m̄ is the mean of m_p over plates of the same kind (with or without positive-control wells);
- y_i ← y_i − α_p (m_p − m̄) for every well on plate p;
- then the mean over all `EMPTY_control` wells is subtracted from every well.

The last re-centring step matters. The scorer uses raw cosine, so the shared offset left by the
strong positive-control phenotypes would otherwise roughly halve recall.

Variants:

| name | α_p |
|---|---|
| no shift | 0 |
| full shift | 1 |
| half shift | ½ |
| empirical Bayes | τ²/(τ² + v_p), below |

**Empirical-Bayes weight.**

- v_p is the sampling variance (trace) of m_p: v_p = RSS_p / ((n_p − G_p) n_p). RSS_p is the
  sum of squared deviations of the plate's control wells from their group (EMPTY / positive) means,
  n_p the number of control wells and G_p the number of groups on the plate.
- τ² is the variance of the true plate offsets: τ² = max(mean_p ‖m_p − m̄‖² − mean_p v_p, 0),
  computed separately for each kind of plate.

The weight uses only control wells, never the benchmark.

### 3.4 Why a partial shift wins

Model one feature direction: a gene well on plate p is y = g + b_p + ε. Here b_p is the plate
offset shared by the plate's genes, with variance τ². The observed shift is s_p = m_p − m̄ = b_p + η_p,
where η_p (variance v) is what the control mean has that the genes don't share: sampling noise and
plate-to-plate variation in the positive-control phenotype.

- **Error after the shift.** Subtracting α s_p leaves e_p = (1 − α) b_p − α η_p on every gene of the
  plate, with E e_p² = (1 − α)² τ² + α² v:
  - τ² with no shift;
  - v with the full shift;
  - (τ² + v)/4 with half.
- **Optimum.** α* = τ²/(τ² + v), with error τ² v/(τ² + v). Half beats both no shift and the full
  shift whenever ⅓ < τ²/v < 3.
- **Why this matters for recall.** e_p is shared by every gene on the plate, so it inflates the
  cosine of same-plate gene pairs. RxRx3-core gene vectors average only a few wells on a few plates,
  so the error doesn't average out.
- **Where the estimate can go wrong.** The empirical-Bayes v_p only sees within-plate sampling
  noise. Plate-to-plate variation of the positive-control effect that the genes don't share is
  counted as offset, so α is overestimated. This is the likely reason it trails half on Phenom-2,
  which has the largest and most consistent positive-control effect.
- **A more general estimator** is the cross-plate regression α̂ = Σ_p ⟨s_p, r_p⟩ / Σ_p ‖s_p‖², with
  r_p the plate's mean over non-control wells minus the global mean. It is not implemented yet.

## 4. Results

### 4.1 Reproducing Table 6 (no shift)

Recall, both tails at 0.95, CORUM:

| set | PCA-CS per experiment | TVN per experiment | paper Table 6 (PCA-CS) |
|---|---|---|---|
| CellProfiler | .361 | .409 | .361 |
| OpenPhenom (released) | .326 | .379 | .300 |
| OpenPhenom (ours, `embed_rxrx3_core.py`) | .322 | .366 | — |
| Phenom-1 | .395 | .459 | .395 |
| Phenom-2 | .479 | .529 | .486 |

All databases for the released sets, PCA-CS, as ours / paper:

| set | CORUM | HuMAP | Reactome | StringDB |
|---|---|---|---|---|
| CellProfiler | .361 / .361 | .438 / .444 | .160 / .160 | .330 / .330 |
| OpenPhenom | .326 / .300 | .388 / .352 | .170 / .158 | .296 / .281 |
| Phenom-1 | .395 / .395 | .486 / .482 | .192 / .188 | .349 / .349 |
| Phenom-2 | .479 / .486 | .552 / .553 | .195 / .197 | .410 / .415 |

- OpenPhenom is the only set that comes out clearly above the paper (by .012–.036).
- TVN per experiment beats PCA-CS on every set for CORUM, HuMAP and StringDB.
- PCA-CS per plate instead of per experiment changes recall by at most about 0.01.

**Our embeddings.** `embed_rxrx3_core.py` splits each 512² six-channel image into four 256 px
crops, scales them to [0, 1], instance-normalises each channel, and embeds them with the OpenPhenom
ViT-S/16 (mean of patch tokens, bf16). Crops are averaged per image, then per well. It recovers
222,596 of the 222,601 released wells, with a median per-well cosine of 0.987 to the released vectors
(1st percentile 0.965).

### 4.2 The control shift

From `rxrx3_core_proxy_minimal_repro.py`. Each cell is recall at the 95 / 99 / 99.5 / 99.9th
percentile, both tails.

**PCA-CS**

CORUM

| set | no shift | full shift | half shift | empirical Bayes |
|---|---|---|---|---|
| OpenPhenom | .326 / .126 / .084 / .034 | .431 / .231 / .176 / .079 | .410 / .184 / .134 / .060 | .428 / .188 / .136 / .063 |
| Phenom-1 | .398 / .174 / .126 / .067 | .467 / .265 / .203 / .094 | .500 / .256 / .194 / .098 | .498 / .259 / .194 / .097 |
| Phenom-2 | .480 / .226 / .174 / .092 | .402 / .231 / .199 / .105 | .529 / .312 / .233 / .115 | .505 / .289 / .222 / .112 |
| CellProfiler | .357 / .165 / .125 / .063 | .422 / .232 / .177 / .078 | .458 / .237 / .188 / .093 | .471 / .235 / .184 / .090 |

HuMAP

| set | no shift | full shift | half shift | empirical Bayes |
|---|---|---|---|---|
| OpenPhenom | .388 / .166 / .111 / .052 | .507 / .271 / .216 / .111 | .486 / .219 / .168 / .085 | .495 / .221 / .170 / .090 |
| Phenom-1 | .489 / .218 / .162 / .101 | .547 / .332 / .264 / .134 | .593 / .320 / .249 / .139 | .585 / .319 / .248 / .138 |
| Phenom-2 | .554 / .281 / .223 / .130 | .452 / .299 / .251 / .148 | .603 / .367 / .287 / .159 | .576 / .340 / .282 / .154 |
| CellProfiler | .437 / .212 / .154 / .088 | .504 / .301 / .223 / .108 | .543 / .297 / .235 / .122 | .545 / .293 / .231 / .118 |

Reactome

| set | no shift | full shift | half shift | empirical Bayes |
|---|---|---|---|---|
| OpenPhenom | .170 / .058 / .046 / .018 | .130 / .042 / .039 / .023 | .156 / .054 / .033 / .016 | .153 / .054 / .033 / .016 |
| Phenom-1 | .192 / .077 / .054 / .030 | .167 / .062 / .051 / .032 | .197 / .083 / .058 / .033 | .186 / .072 / .056 / .033 |
| Phenom-2 | .195 / .093 / .072 / .037 | .176 / .070 / .054 / .039 | .192 / .093 / .065 / .039 | .188 / .090 / .063 / .039 |
| CellProfiler | .158 / .060 / .044 / .025 | .141 / .053 / .039 / .025 | .170 / .062 / .039 / .030 | .162 / .051 / .037 / .026 |

StringDB

| set | no shift | full shift | half shift | empirical Bayes |
|---|---|---|---|---|
| OpenPhenom | .297 / .128 / .087 / .041 | .341 / .187 / .149 / .071 | .341 / .161 / .120 / .055 | .347 / .163 / .121 / .058 |
| Phenom-1 | .350 / .163 / .121 / .069 | .386 / .221 / .176 / .086 | .416 / .227 / .175 / .089 | .410 / .227 / .174 / .088 |
| Phenom-2 | .411 / .215 / .165 / .089 | .352 / .204 / .172 / .099 | .432 / .267 / .210 / .107 | .413 / .252 / .202 / .104 |
| CellProfiler | .329 / .156 / .112 / .059 | .354 / .195 / .147 / .069 | .383 / .204 / .160 / .079 | .379 / .197 / .158 / .075 |

**TVN**

CORUM

| set | no shift | full shift | half shift | empirical Bayes |
|---|---|---|---|---|
| OpenPhenom | .380 / .173 / .130 / .054 | .465 / .244 / .188 / .080 | .466 / .220 / .173 / .075 | .478 / .226 / .173 / .076 |
| Phenom-1 | .459 / .205 / .157 / .080 | .485 / .286 / .205 / .089 | .542 / .299 / .225 / .105 | .537 / .293 / .217 / .104 |
| Phenom-2 | .530 / .268 / .213 / .104 | .408 / .238 / .196 / .095 | .560 / .334 / .256 / .125 | .521 / .313 / .237 / .123 |
| CellProfiler | .409 / .206 / .157 / .083 | .435 / .237 / .174 / .067 | .492 / .269 / .208 / .108 | .503 / .266 / .203 / .105 |

HuMAP

| set | no shift | full shift | half shift | empirical Bayes |
|---|---|---|---|---|
| OpenPhenom | .445 / .216 / .162 / .074 | .509 / .282 / .223 / .111 | .518 / .266 / .204 / .103 | .521 / .264 / .203 / .104 |
| Phenom-1 | .554 / .273 / .205 / .115 | .548 / .356 / .265 / .126 | .622 / .362 / .286 / .143 | .615 / .349 / .273 / .142 |
| Phenom-2 | .613 / .327 / .264 / .143 | .444 / .302 / .246 / .134 | .629 / .388 / .308 / .167 | .578 / .361 / .289 / .164 |
| CellProfiler | .527 / .260 / .198 / .109 | .516 / .301 / .221 / .091 | .573 / .334 / .265 / .139 | .569 / .329 / .257 / .136 |

Reactome

| set | no shift | full shift | half shift | empirical Bayes |
|---|---|---|---|---|
| OpenPhenom | .158 / .060 / .039 / .019 | .130 / .042 / .040 / .023 | .156 / .054 / .040 / .023 | .149 / .053 / .039 / .023 |
| Phenom-1 | .192 / .076 / .058 / .030 | .155 / .067 / .049 / .030 | .192 / .084 / .054 / .033 | .185 / .083 / .056 / .033 |
| Phenom-2 | .204 / .107 / .072 / .039 | .170 / .074 / .056 / .037 | .193 / .091 / .069 / .040 | .188 / .091 / .065 / .039 |
| CellProfiler | .167 / .063 / .047 / .025 | .130 / .060 / .040 / .025 | .172 / .063 / .046 / .030 | .169 / .062 / .042 / .028 |

StringDB

| set | no shift | full shift | half shift | empirical Bayes |
|---|---|---|---|---|
| OpenPhenom | .314 / .161 / .123 / .054 | .351 / .196 / .156 / .071 | .368 / .185 / .144 / .069 | .365 / .186 / .145 / .068 |
| Phenom-1 | .384 / .196 / .148 / .077 | .390 / .234 / .178 / .080 | .432 / .248 / .199 / .095 | .419 / .242 / .191 / .094 |
| Phenom-2 | .438 / .249 / .196 / .097 | .349 / .207 / .167 / .088 | .447 / .280 / .223 / .114 | .417 / .269 / .209 / .112 |
| CellProfiler | .371 / .185 / .142 / .071 | .367 / .203 / .145 / .058 | .404 / .234 / .181 / .089 | .404 / .230 / .176 / .087 |

**Consistency.** Over CORUM, HuMAP and StringDB × 4 sets × 4 cutoffs (48 cells per alignment):

| method | PCA-CS: beats no shift | PCA-CS: mean gain | TVN: beats no shift | TVN: mean gain |
|---|---|---|---|---|
| full shift | 44 / 48 | +0.044 | 30 / 48 | +0.011 |
| half shift | 48 / 48 | +0.057 | 48 / 48 | +0.045 |
| empirical Bayes | 48 / 48 | +0.054 | 45 / 48 | +0.038 |

- **Empirical Bayes vs half.** The mean difference (empirical Bayes − half) is within ±0.007 for
  every set and alignment except Phenom-2: −0.014 after PCA-CS and −0.020 after TVN. Its three
  losses to no shift are all Phenom-2 TVN at the 95th percentile.
- **Where the full shift is best.** Only for OpenPhenom at the 99th percentile and above. The
  weakest embedding apparently keeps the largest plate offset after alignment, so α* is nearer 1.
- **Reactome.** No variant helps; the full shift costs up to 0.04.
- **Positive-control genes don't explain the gain.** Leaving the positive-control genes (MTOR,
  PLK1, ...) out of the scored set (`rxrx3_core_proxy.py --exclude-positives`) changes every number
  by at most +0.01. So the gain isn't leakage from scoring the proxies themselves.

**Estimated weights.** The median α̂ per plate is:

| alignment | OpenPhenom | Phenom-1 | Phenom-2 | CellProfiler |
|---|---|---|---|---|
| PCA-CS | .61 | .57 | .44 | .48 |
| TVN | .56 | .52 | .44 | .44 |
| EFAAR | .87 | .86 | .81 | .97 |

The minimum is 0, on plate `gene-051/5`.

### 4.3 EFAAR and the shift

From `rxrx3_core_shrinkage.py`; CORUM, both tails, 95 / 99 / 99.5 / 99.9:

| set | no shift | full shift | half shift | empirical Bayes |
|---|---|---|---|---|
| OpenPhenom | .323 / .145 / .089 / .020 | .401 / .203 / .154 / .049 | .396 / .193 / .147 / .045 | .402 / .209 / .155 / .053 |
| Phenom-1 | .382 / .156 / .095 / .027 | .449 / .224 / .166 / .065 | .469 / .227 / .175 / .080 | .461 / .232 / .180 / .089 |
| Phenom-2 | .442 / .196 / .136 / .065 | .467 / .287 / .203 / .095 | .508 / .280 / .207 / .105 | .480 / .294 / .215 / .109 |
| CellProfiler | .385 / .164 / .103 / .035 | .152 / .045 / .025 / .003 | .266 / .132 / .085 / .041 | .419 / .228 / .171 / .061 |

- **EFAAR alone is worse than TVN per experiment on RxRx3-core.** A core plate keeps only about
  80 wells, about 16 of them negatives, which is too few for a per-plate covariance.
- **Under EFAAR every plate's EMPTY mean is already 0.** So m_p − m̄ is just the positive-control
  deviation, scaled by about 12/28.
- **CellProfiler EFAAR + full shift is a numerical blow-up.**
  - Three plates have shifts over 10× the median (max 3.6e4 vs a median of 39, on plate
    `gene-168/5`), driven by a few features.
  - The cause: EFAAR scales each feature by 1/std over the plate's 16 negative wells, so a feature
    that is nearly constant there explodes on the positive wells.
  - Empirical Bayes gives those plates small weights.

### 4.4 Other shift variants (`rxrx3_core_shift_diagnostics.py`)

CORUM, both tails, 95 / 99 / 99.5 / 99.9:

| set | map | none | full shift | half | median |
|---|---|---|---|---|---|
| OpenPhenom | TVN | .379 / .173 / .130 / .054 | **.465 / .245 / .186 / .080** | .467 / .220 / .173 / .074 | .449 / .206 / .160 / .070 |
| OpenPhenom | EFAAR | .323 / .145 / .089 / .020 | **.401 / .203 / .154 / .049** | .396 / .193 / .147 / .045 | .340 / .155 / .108 / .025 |
| Phenom-1 | TVN | .459 / .205 / .157 / .080 | .486 / .286 / .205 / .089 | **.542 / .298 / .225 / .105** | .510 / .261 / .189 / .095 |
| Phenom-1 | EFAAR | .382 / .156 / .095 / .027 | .449 / .224 / .166 / .065 | **.470 / .227 / .175 / .080** | .392 / .168 / .108 / .036 |
| Phenom-2 | TVN | .529 / .268 / .213 / .104 | .409 / .240 / .199 / .095 | **.561 / .335 / .256 / .125** | .555 / .308 / .233 / .116 |
| Phenom-2 | EFAAR | .442 / .196 / .136 / .065 | .467 / .287 / .203 / .095 | **.508 / .280 / .207 / .105** | .451 / .201 / .147 / .070 |
| CellProfiler | TVN | .409 / .204 / .157 / .083 | .437 / .237 / .174 / .067 | **.494 / .270 / .207 / .108** | .455 / .237 / .189 / .094 |
| CellProfiler | EFAAR | .385 / .164 / .103 / .035 | .152 / .045 / .025 / .003 | .268 / .132 / .085 / .041 | **.392 / .170 / .120 / .041** |

- **Negatives only equals no shift**, because TVN and EFAAR already centre on the negatives.
  **Positives only is the worst variant everywhere.**
- **Phenom-2 has the most consistent positive-control effect across plates:** a cosine of .69 to
  its mean, against .54 for Phenom-1. Its full shift is also the largest relative to the
  negative-well spread (21.5 vs 14.4). So its residual plate offset is likely small, and the full
  shift is mostly noise there.
- **By database** (share of the 32 set × map × cutoff cells where the first variant beats the
  second):

  | | CORUM | HuMAP | StringDB | Reactome | SIGNOR |
  |---|---|---|---|---|---|
  | half > none | .91 | .91 | .91 | .62 | .16 |
  | half > full | .75 | .81 | .81 | .88 | .75 |
  | full > none | .72 | .62 | .66 | .09 | .09 |

## 5. Scripts

All of these are in `scripts/`. They read the data from `outputs/scratch/rxrx3_core/` and write to
`outputs/2026-11-15/`; both folders are hard-coded. Everything except the embedding script runs
on CPU only.

| script | what it does | run time |
|---|---|---|
| `rxrx3_core_minimal_repro.py` | Self-contained Table 6 reproduction: PCA-CS and TVN, 95th percentile, against the paper | under 1 h |
| `rxrx3_core_proxy_minimal_repro.py` | Self-contained shift results (Section 4.2): every set × PCA-CS / TVN × four shifts × four cutoffs, plus tables and two figures | 1 h on 32 CPUs (one 16-CPU run took over 4 h) |
| `embed_rxrx3_core.py` | Our OpenPhenom embeddings of the core images → `rxrx3_core/ours_openphenom.parquet` | 3.5 h, one GPU |
| `rxrx3_core_repro.py [extra.parquet ...]` | Official alignments (PCA-CS per experiment and per plate, TVN, none) on every released set plus any extra ones → `rxrx3_core_repro.csv` | under 1 h |
| `rxrx3_core_proxy.py [sets] [--exclude-positives]` | PCA-CS / TVN / EFAAR, each with and without the full shift, at every cutoff and tail → `rxrx3_core_proxy[_<set>][_nopos].csv`. Sets: `ours` (default), `released` (= OpenPhenom), `CellProfiler`, `Phenom-1`, `Phenom-2` | 13–36 min per set |
| `rxrx3_core_shrinkage.py <set>` | none / full / α = 1 / α = ½ / empirical Bayes after PCA-CS, TVN and EFAAR, and prints the α̂ quantiles → `rxrx3_core_shrinkage_<set>.csv` | 17–50 min |
| `rxrx3_core_shift_diagnostics.py <set> <TVN\|EFAAR\|PCA-CS>` | Shift size against gene spread, positive-control consistency, outlier plates; variants none / full / negatives only / positives only / median / half → `shift_diagnostics_<set>_<map>.csv` | under 1 h |
| `plot_shift_deltas.py` | Absolute recall per shift, from the shrinkage CSVs → `shift_deltas_<PCA-CS\|TVN>.png`, `shift_deltas.csv` | seconds |
| `plot_shift_variants.py` | Variant gains over no shift, from the diagnostics CSVs (TVN and EFAAR, all four sets) → `shift_variants.png` | seconds |

The scripts that load all four sets at once need up to about 190 GB of RAM. To reproduce
everything in order:

```bash
uv run python scripts/embed_rxrx3_core.py
```

```bash
uv run python scripts/rxrx3_core_repro.py ours_openphenom.parquet
```

```bash
for s in released Phenom-1 Phenom-2 CellProfiler; do uv run python scripts/rxrx3_core_shrinkage.py $s; done
```

```bash
for s in released Phenom-1 Phenom-2 CellProfiler; do for m in TVN EFAAR; do uv run python scripts/rxrx3_core_shift_diagnostics.py $s $m; done; done
```

```bash
uv run python scripts/plot_shift_deltas.py && uv run python scripts/plot_shift_variants.py
```

## 6. Pitfalls and open questions

**Pitfalls that change the numbers:**

- **The positive-control label.** Use `well_type_label == "CRISPR Gene Positive Controls"`.
  `treatment == "CRISPR_control"` occurs only on compound plates, so using it gives a
  negatives-only shift, which does nothing.
- **Re-centring.** Without the final re-centring on the negatives, the shared offset from the
  strong positive controls roughly halves recall under the raw-cosine scorer.
- **The two shift implementations differ slightly.** `rxrx3_core_proxy.control_shift` (the
  original full shift) subtracts m_p rather than m_p − m̄. It differs from α = 1 in
  `shrunk_shift` and in the minimal script by at most 0.0025 recall.

**Open questions:**

- **How to set α** with the cross-plate regression estimator (Section 3.4), which sees what the
  genes actually share. Its potential matrix form is T (T + V_p)⁻¹ in place of a scalar.
- **A median control mean.** It is robust to outlier wells, and is the best map for CellProfiler
  EFAAR and for Reactome / SIGNOR on most rows.
- **Why Reactome and SIGNOR don't benefit.** Pathway-level relationships may be carried partly by
  the very plate structure the shift removes.
- **The chromosome-arm correction.** The paper applies it on full RxRx3, but it isn't applied
  here.
