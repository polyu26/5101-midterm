# Cat Image Arena — Analysis Pipeline

A dataset of pairwise cat-image preferences for data analysis and preference prediction, plus a five-notebook analysis pipeline (Colab-ready).

## Project objectives

- **Explore and interpret the data:** examine human preferences, disagreement between annotators, and relationships between annotator characteristics and image preferences.
- **Predict preferences:** use the training data to predict each annotator’s image preference in the test data. In `test.csv`, fill in `prob_a` with the predicted probability of choosing image A (between 0 and 1). Fill in `choice` with `A` if `prob_a ≥ 0.5`, or `B` otherwise.

Each row represents one annotator's response to one image pair. A pair may have multiple responses from different annotators. There is **no respondent ID** — the same person may appear more than once without us knowing.

---

## Dataset files

| File | Description |
| --- | --- |
| `train.csv` | 24,896 responses with observed `choice` values. |
| `test.csv` | 6,224 test rows with `choice` and `prob_a` left blank. |
| `images/` | 1,558 images organized into cat folders. |

### Image paths

Get the images from [Hugging Face](https://huggingface.co/datasets/missing12/cat-image-arena-images/tree/main/images).

The `a_image` and `b_image` fields contain paths relative to `images/`. For example:

```text
Table value: cat_000002/cat_000002_0001.jpg
Local file:  images/cat_000002/cat_000002_0001.jpg
```

### Fields

| Field | Meaning |
| --- | --- |
| `train_row_id` | Unique training-row identifier; present only in the training file. |
| `test_row_id` | Unique test-row identifier; present only in the test files. |
| `a_cat`, `b_cat` | Cat identifiers for sides A and B. |
| `a_image`, `b_image` | Image paths for sides A and B. |
| `choice` | Observed preference: `A` or `B`. Withheld in the test files. |
| `prob_a` | Predicted probability of choosing image A, between 0 and 1; present only in the test file. |
| `comparison_type` | `within_cat`: different photos of the same cat; `between_cats`: photos of different cats. |
| `gender` | Self-reported gender. |
| `pet_history` | Household pet ownership: `current`, `past`, or `never`. |
| `pet_care` | Personal pet-care experience: `never`, `occasionally`, `past_regular`, or `current_regular`. |
| `pet_contact` | Frequency of interaction with pets: `never`, `rarely`, `sometimes`, `often`, or `daily`. |
| `future_pet` | Intention to own or continue owning pets: `yes`, `no`, or `unsure`. |
| `mbti_ie` | `I` or `E`, derived from self-reported MBTI; `unknown` means unsure or not tested. |

---

## Notebook pipeline

All notebooks are written to run in **Google Colab** or locally. They share a common path-finding pattern (look for CSVs in the working directory, `artifacts/`, Drive mounts, etc.) and write intermediate tables under a `prepared/` folder.

```text
train.csv
    │
    ▼
01_clean_training_responses_colab.ipynb
    │  → prepared/cleaned_training_responses.csv
    │
    ├──────────────────┬──────────────────────────┐
    ▼                  ▼                          ▼
02_agreement…      03_demographic…           04_image_features…
    │                  │                          │
    │  side-adjusted   │  demographic EDA +       │  image embeddings
    │  cat/image       │  logistic baseline       │  & simple features
    │  scores          │                          │
    └──────────────────┴────────────┬─────────────┘
                                   ▼
                    05_preference_models_colab.ipynb
                         (BT, RF, image rankers)
```

### 01 — Clean training responses

**File:** `01_clean_training_responses_colab.ipynb`

**Purpose:** Turn the raw training file into a trustworthy analysis table and answer basic questions about coverage, choice balance, respondent background, and simple agreement patterns.

| | |
| --- | --- |
| **Input** | `train.csv` (24,896 rows) |
| **Output** | `prepared/cleaned_training_responses.csv` |
| **Key steps** | Load → missing-value check → cleaning & validation → coverage / frequency of cats & photos → choice balance → background fields → repeated pairs → first look at agreement |

**Derived columns of note:** `chose_a`, `demo_complete`, ordered/unordered pair IDs. These feed every later notebook.

**Main findings (high level):** Keep all 24,896 rows; missing background is an all-or-nothing block of 508 rows; mild side-A preference (~52%); for between-cat pairs, agreement rises when the two cats differ more in win rate.

---

### 02 — Agreement, reliability and side bias

**File:** `02_agreement_and_reliability_colab.ipynb`

**Purpose:** Measure whether choices are reproducible and associated with measurable structure. No predictor model is fit — only descriptive reliability and association measures.

| | |
| --- | --- |
| **Input** | `prepared/cleaned_training_responses.csv` |
| **Outputs** | `prepared/side_adjusted_cat_scores_between.csv`, related side-adjusted image/cat scores |

**What it measures:**
- Side-A rate stability on repeated pairs
- Agreement on the same ordered pair (Cohen’s κ)
- Effect of swapping the two photos (photo signal vs side signal)
- Split-half reliability of image and cat win rates
- Association of background fields with the side chosen (Cramér’s V)
- Whether the 508 blank-background rows differ in choice rate

**Main findings (high level):** A single response is noisy (κ ≈ 0.11); preference still carries a real photo/cat signal (negative swap correlation; cat split-half *r* ≈ 0.69); demographics barely move P(A).

---

### 03 — Demographic preference EDA

**File:** `03_demographic_preference_eda_colab.ipynb`

**Purpose:** Describe the sample and preference patterns in detail so counts (not only summary statistics) are visible. Builds on notebook 2’s finding that background fields are only weakly associated with choice.

| | |
| --- | --- |
| **Input** | `prepared/cleaned_training_responses.csv` |
| **Outputs** | Tables / plots only (no required CSV for downstream steps) |

**What it covers:**
- Response counts by choice, comparison type, and every background field
- Joint demographic profiles
- Preference for side A by each characteristic (Wilson 95% CIs)
- Preference gaps × comparison type
- Simple logistic-regression baseline on demographics + comparison type

**Important:** All counts are **numbers of responses**, not unique people.

**Main findings (high level):** Sample is uneven across demographics; P(A) varies only a few points across groups (largest span ≈ 5 points for MBTI). Use these tables for description, not as strong predictors.

---

### 04 — Image feature extraction

**File:** `04_image_features_colab (1).ipynb`

**Purpose:** Download cat photos from Hugging Face and extract visual features that can feed a Bradley–Terry / ranking model (shared scorer on item features).

| | |
| --- | --- |
| **Input** | `prepared/cleaned_training_responses.csv` + images from Hugging Face |
| **Outputs** | `prepared/image_features_resnet.csv` (512-d ResNet18, L2-normalised)<br>`prepared/image_features_simple.csv` (colour, brightness, contrast, …)<br>`prepared/image_feature_index.csv`<br>optional `prepared/cat_features_resnet_mean.csv` |

**Runtime:** CPU is fine (~5–15 min for ~1.3k images). GPU is faster but not required.

---

### 05 — Preference models

**File:** `05_preference_models_colab.ipynb`

**Purpose:** Fit course-aligned statistical models and evaluate them with **pair-aware** train/validation splits. Preprocessing is fit on the training fold only.

| | |
| --- | --- |
| **Required input** | `prepared/cleaned_training_responses.csv` |
| **Optional inputs** | Side-adjusted scores (notebook 02), image features (notebook 04) |

**Models:**
1. **Logistic regression** on demographics + comparison type
2. **Bradley–Terry (BT)** ranking of cats (logistic regression on score differences)
3. **Random forest** baseline on the same tabular features
4. **Image-feature ranking** on Δx = xᵢ − xⱼ (interpretable features and/or ResNet embeddings)
5. Optional hybrids (BT score + demographics, image features + demographics)

**Evaluation:** Pair-aware split so repeated comparisons cannot leak across folds. Metrics reported on the held-out fold only.

**Expected takeaway:** Demographics alone are weak; a cat-level BT model captures the replicable preference signal found in notebook 02; image features help especially on within-cat pairs where a single cat score cannot explain differences.

---

## Intermediate / prepared CSVs

Produced under a `prepared/` directory (created by the notebooks):

| File | Produced by | Description |
| --- | --- | --- |
| `cleaned_training_responses.csv` | 01 | Validated training table with derived columns (`chose_a`, `demo_complete`, pair IDs, …) |
| `side_adjusted_cat_scores_between.csv` (and related) | 02 | Empirical side-adjusted strengths for cats/images |
| `image_features_resnet.csv` | 04 | 512-d ResNet18 embedding per image |
| `image_features_simple.csv` | 04 | Hand-crafted interpretable image features |
| `image_feature_index.csv` | 04 | Map from image path → feature row index |
| `cat_features_resnet_mean.csv` | 04 (optional) | Mean ResNet embedding per cat |

---

## How to run

1. Place `train.csv` (and optionally `test.csv`) where the notebooks can find them (working directory, `artifacts/`, or Colab Drive mount).
2. Run notebooks **in order** 01 → 02 / 03 / 04 → 05. Notebooks 02, 03 and 04 all depend only on the cleaned table from 01; 05 can use outputs from 02 and 04 when available.
3. For image features (notebook 04), ensure internet access so images can be downloaded from Hugging Face, or place the `images/` tree locally.

Each notebook begins with a setup cell that locates data and creates the `prepared/` folder.

---

## Course / report context

These notebooks support the midterm report sections on data understanding & preparation, exploratory analysis, methods, results, and interpretation. Models and diagnostics are aligned with course topics (logistic regression, tree ensembles, Bradley–Terry ranking / shared scorers).

For research and educational use only.
