# Amazon ML Challenge 2026 --- Business Entity Resolution

End-to-end starter pipeline for the **Amazon ML Challenge 2026 Business
Entity Resolution** problem.

The project matches records from Source 2 (S2) and Source 3 (S3) to
records in the deduplicated Source 1 (S1), using business name, address,
country, and derived similarity signals.

The pipeline is designed for the actual large-scale dataset and keeps
the raw dataset, intermediate artifacts, and final competition outputs
separate.

------------------------------------------------------------------------

## 1. Problem Overview

The challenge is an **entity-resolution / record-linkage** problem.

For every S1 entity, the task is to identify the corresponding entity
IDs from S2 and/or S3.

A single S1 entity can have:

-   no matches
-   one match
-   multiple matches

The training ground truth contains:

``` text
source1_entity_id    matched_entity_ids
```

where `matched_entity_ids` is a comma-separated list. An empty value
means that the S1 entity has no match.

### Important dataset facts already observed

Training:

-   S1: 2,206,821 rows
-   S2: 5,034,616 rows
-   S3: 5,285,603 rows
-   Total training source records: \~12.5M

Test:

-   S1: 1,732,544 rows
-   S2: 4,887,273 rows
-   S3: 5,082,316 rows
-   Total test source records: \~11.7M

Ground truth:

-   S1 entities: 2,206,821
-   Singleton/no-match S1 entities: 123,247 (\~5.59%)
-   Non-singleton S1 entities: 2,083,574
-   Total true matched IDs: 7,638,365
-   Maximum matches for one S1: 11
-   No duplicate S1 IDs were observed
-   No duplicate matched S2/S3 IDs were observed

Countries observed:

-   Training: US, India
-   Test: US, India, France

**Do not hard-code the training country set. France exists in test.**

------------------------------------------------------------------------

# 2. Repository Structure

The recommended project structure is:

``` text
AMAZON ML CHALLENGE/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── src/
│   ├── pipeline.py
│   └── make_synthetic.py
│
├── 6ab10eb3b23ba_student_resource/
│   └── student_resource/
│       ├── dataset/
│       │   ├── train/
│       │   │   ├── train_source1.tsv
│       │   │   ├── train_source2.tsv
│       │   │   ├── train_source3.tsv
│       │   │   └── train_ground_truth.tsv
│       │   │
│       │   ├── test/
│       │   │   ├── test_source1.tsv
│       │   │   ├── test_source2.tsv
│       │   │   └── test_source3.tsv
│       │   │
│       │   └── [organizer files]
│       │
│       ├── utils/
│       │   └── validate_submission.py
│       │
│       ├── Documentation_template.md
│       └── README.md
│
├── artifacts/
│   ├── normalized/
│   ├── indexes/
│   ├── candidates/
│   ├── features/
│   ├── models/
│   └── validation/
│
└── output/
    ├── matching_results.tsv
    └── candidate_pairs.tsv
```

### What each directory means

### `dataset/`

Raw competition data only.

**Do not modify the original TSV files.**

This directory should be ignored by Git.

### `src/`

All competition code.

The main implementation is intentionally orchestrated through:

``` text
src/pipeline.py
```

### `artifacts/`

Intermediate/generated data.

Examples:

-   normalized records
-   indexes
-   validation splits
-   candidate pairs
-   features
-   trained models
-   validation results
-   thresholds

These files allow the pipeline to resume without rebuilding everything
from scratch.

### `output/`

Only final competition deliverables:

``` text
matching_results.tsv
candidate_pairs.tsv
```

------------------------------------------------------------------------

# 3. Download the Competition Dataset

The real dataset is provided separately by the organizers.

## Dataset Drive Link

**Paste the official dataset Google Drive link here:**

> `[PASTE OFFICIAL DATASET DRIVE LINK HERE]`

Download the dataset ZIP from the official Drive link.

### Important

Do not upload the multi-GB raw dataset to GitHub.

The dataset is local input data and should remain inside the ignored
`dataset/` directory.

------------------------------------------------------------------------

# 4. Download and Extract the Dataset

After downloading the ZIP:

1.  Open the downloaded ZIP/archive.
2.  Extract the organizer's `student_resource` directory.
3.  Place it inside the project.
4.  Confirm that the following folders exist:

``` text
student_resource/
└── dataset/
    ├── train/
    └── test/
```

The training directory must contain:

``` text
train_source1.tsv
train_source2.tsv
train_source3.tsv
train_ground_truth.tsv
```

The test directory must contain:

``` text
test_source1.tsv
test_source2.tsv
test_source3.tsv
```

### Do not accidentally create this:

``` text
dataset/dataset/train/
```

The expected location must be:

``` text
student_resource/dataset/train/
```

------------------------------------------------------------------------

# 5. Windows PowerShell Setup

This project is being developed on Windows.

Open PowerShell and go to the project root:

``` powershell
cd "C:\Users\<YOUR_USERNAME>\OneDrive\Desktop\Amazon ML Challenge"
```

Check the directory:

``` powershell
dir
```

You should see:

``` text
src
6ab10eb3b23ba_student_resource
README.md
.gitignore
```

------------------------------------------------------------------------

# 6. Create a Virtual Environment

Recommended:

``` powershell
python -m venv .venv
```

Activate it:

``` powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use:

``` powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

and then:

``` powershell
.\.venv\Scripts\Activate.ps1
```

Verify Python:

``` powershell
python --version
```

------------------------------------------------------------------------

# 7. Install Dependencies

From the project root:

``` powershell
pip install -r requirements.txt
```

If you want to confirm the environment:

``` powershell
pip list
```

The exact dependency versions should be controlled by
`requirements.txt`.

------------------------------------------------------------------------

# 8. Verify the Dataset Before Running ML

Because the real dataset contains millions of rows, first verify that
the files exist.

From:

``` text
...\student_resource\dataset
```

run:

``` powershell
dir train
dir test
```

You should see the expected TSV files.

------------------------------------------------------------------------

# 9. Check Dataset Dimensions

From the directory:

``` text
...\student_resource\dataset
```

run:

``` powershell
python -c "import pandas as pd; files=['train/train_source1.tsv','train/train_source2.tsv','train/train_source3.tsv','train/train_ground_truth.tsv','test/test_source1.tsv','test/test_source2.tsv','test/test_source3.tsv']; [(print(f, pd.read_csv(f, sep='\t').shape, pd.read_csv(f, sep='\t').columns.tolist())) for f in files]"
```

Expected training/test scale is approximately:

``` text
train_source1.tsv       2,206,821 x 4
train_source2.tsv       5,034,616 x 4
train_source3.tsv       5,285,603 x 4
train_ground_truth.tsv  2,206,821 x 2

test_source1.tsv        1,732,544 x 4
test_source2.tsv        4,887,273 x 4
test_source3.tsv        5,082,316 x 4
```

The source files contain:

``` text
entity_id
business_name
business_address
country
```

The ground truth contains:

``` text
source1_entity_id
matched_entity_ids
```

------------------------------------------------------------------------

# 10. Run the Ground-Truth Audit

Before training anything, inspect the structure of the matching problem.

Go to:

``` text
...\student_resource\dataset
```

Run:

``` powershell
python -c "import pandas as pd; gt=pd.read_csv('train/train_ground_truth.tsv',sep='\t',dtype=str,keep_default_na=False); m=gt['matched_entity_ids'].str.strip(); c=m.eq('').sum(); counts=m[m!=''].str.split(',').map(len); print('GROUND TRUTH'); print('Rows:',len(gt)); print('Singleton S1:',c); print('Non-singleton S1:',len(gt)-c); print('Match-count distribution:'); print(counts.value_counts().sort_index().head(30).to_string()); print('Max matches:',counts.max() if len(counts) else 0); allm=[x for s in m[m!=''] for x in s.split(',')]; print('S1 unique:',gt.source1_entity_id.nunique()); print('Duplicate S1 rows:',gt.source1_entity_id.duplicated().sum()); print('Total matched IDs:',len(allm)); print('Unique matched IDs:',len(set(allm))); print('Matched-ID duplicates:',len(allm)-len(set(allm))); print('S2 matches:',sum(x.startswith('S2-') for x in allm)); print('S3 matches:',sum(x.startswith('S3-') for x in allm));"
```

This checks:

-   singleton/no-match rate
-   match cardinality
-   maximum number of matches
-   duplicate S1 IDs
-   duplicate matched IDs
-   S2/S3 distribution

------------------------------------------------------------------------

# 11. Run the Source-Level Audit

Run:

``` powershell
python -c "import pandas as pd; files=['train/train_source1.tsv','train/train_source2.tsv','train/train_source3.tsv','test/test_source1.tsv','test/test_source2.tsv','test/test_source3.tsv']; [print('\n',f,'\nrows=',len(df),'\nduplicate_ids=',df.entity_id.duplicated().sum(),'\ncountry=',df.country.value_counts(dropna=False).to_dict(),'\nmissing=',df.isna().sum().to_dict()) for f in files for df in [pd.read_csv(f,sep='\t',usecols=['entity_id','country'])]]"
```

This verifies:

-   row count
-   duplicate IDs
-   country distribution
-   missing values

The audit should be performed before making assumptions in the blocking
strategy.

------------------------------------------------------------------------

# 12. Understand the Core Pipeline

The complete ML workflow is:

``` text
RAW TSV
   ↓
DATA AUDIT
   ↓
NORMALIZATION
   ↓
GROUND-TRUTH RESTRUCTURING
   ↓
LEAKAGE-SAFE VALIDATION SPLIT
   ↓
BLOCKING
   ↓
CANDIDATE GENERATION
   ↓
CANDIDATE RECALL EVALUATION
   ↓
PAIR FEATURE ENGINEERING
   ↓
MATCHING MODEL
   ↓
F0.5 THRESHOLD CALIBRATION
   ↓
SINGLETON / DECISION HANDLING
   ↓
FINAL CANDIDATE SET
   ↓
FINAL MATCH PREDICTIONS
   ↓
candidate_pairs.tsv
   ↓
matching_results.tsv
   ↓
OFFICIAL VALIDATOR
```

------------------------------------------------------------------------

# 13. Phase 1 --- Data Loading

`pipeline.py` loads the source files.

Because the dataset contains approximately 12.5M training records and
11.7M test records, avoid repeatedly loading the same TSVs.

The production pipeline should use:

-   controlled dtypes
-   chunked processing where appropriate
-   deterministic processing
-   cached artifacts
-   persistent indexes
-   minimal duplicate loading

The raw TSVs should remain untouched.

------------------------------------------------------------------------

# 14. Phase 2 --- Normalization

`normalize.py` handles:

``` text
business_name
business_address
country
```

The normalization layer should preserve:

1.  the original/raw field
2.  the normalized representation
3.  expanded/derived representations where useful

Normalization exists because entity records can differ in:

-   punctuation
-   capitalization
-   spacing
-   abbreviations
-   formatting
-   address representation
-   token order

Do not discard the raw data.

------------------------------------------------------------------------

# 15. Phase 3 --- Blocking

Blocking is the most important scalability component.

The theoretical comparison space is enormous:

``` text
S1 × S2
+
S1 × S3
```

With the observed dataset scale this is roughly:

``` text
22.8 trillion
```

possible S1-S2/S3 comparisons.

Therefore, **all-pairs matching is not feasible**.

The pipeline instead creates a much smaller candidate set.

Current blocking strategy includes:

-   country
-   name tokens
-   address tokens
-   exact normalized name
-   postal code
-   union of multiple blocking rules

Conceptually:

``` text
S1 record
   ↓
multiple blocking rules
   ↓
union of candidates
   ↓
candidate_pairs
```

### Important candidate-recall principle

A true match that is discarded during blocking can never be recovered by
the downstream ML model.

Therefore:

``` text
Candidate Recall =
True matches present in candidate set
--------------------------------------
Total true matches
```

Candidate recall must be measured before optimizing the pair classifier.

------------------------------------------------------------------------

# 16. Country Blocking

The starter pipeline currently uses country as a precision boost.

However, this assumption must be checked against the real data.

If country labels are noisy or missing, the country restriction can
remove genuine matches.

If validation shows that country intersection is harmful, modify the
blocking logic so country becomes one blocking signal rather than a
mandatory intersection.

**Never hard-code only:**

``` text
US
India
```

because France exists in the test dataset.

------------------------------------------------------------------------

# 17. Blocking Stopwords

`STOPWORD_TOKENS` contains a small manual stopword list.

On the real dataset, inspect high-frequency tokens.

Common tokens such as:

``` text
city names
generic business words
common address terms
```

can create huge candidate sets.

Extend the stopword list only after inspecting actual candidate-volume
statistics.

Do not blindly add words without measuring their effect on recall and
candidate volume.

------------------------------------------------------------------------

# 18. Phase 4 --- Candidate Generation

The final candidate set is written to:

``` text
output/candidate_pairs.tsv
```

This is not merely a diagnostic file.

The competition requirement means the final candidate set should
correspond to the candidates actually passed into the matching model.

Therefore:

``` text
Every final predicted match
MUST EXIST
inside candidate_pairs.tsv
```

Candidate generation must balance:

``` text
high recall
+
manageable candidate volume
```

------------------------------------------------------------------------

# 19. Phase 5 --- Pair Feature Engineering

`features.py` generates pairwise features for:

``` text
S1 ↔ S2
S1 ↔ S3
```

The starter design contains approximately 20 similarity features.

Feature families include:

### Name

-   Levenshtein similarity
-   token sort ratio
-   token set ratio
-   Jaccard similarity
-   character n-gram TF-IDF cosine similarity

### Address

-   Levenshtein similarity
-   token similarity
-   Jaccard similarity
-   character n-gram similarity
-   postal-code match
-   house-number match

### Country

-   exact country match

### Cross-field signals

Examples include:

-   name/address agreement
-   exact normalized name
-   combinations of strong name and address signals

The exact feature implementation should be validated on the real data.

------------------------------------------------------------------------

# 20. Phase 6 --- Training Data

The starter pipeline initially treats every generated candidate that is
not a true match as a negative.

That is acceptable for a synthetic smoke test but is problematic at real
scale because the candidate set can still be highly imbalanced.

The real pipeline should therefore introduce **hard-negative mining**.

Important hard-negative examples include:

``` text
same/similar business name
+
different address
```

and other high-similarity but incorrect candidates.

The goal is to train the matcher on difficult cases rather than
overwhelming it with easy negatives.

------------------------------------------------------------------------

# 21. Phase 7 --- Matching Model

The starter model uses:

``` text
XGBoost classifier
```

The model receives pairwise similarity features and predicts whether a
candidate pair is a genuine match.

The initial model can remain XGBoost because it is practical for
structured pairwise features.

If an embedding/neural component is added later, verify that it
satisfies the competition's stated model-size and license constraints.

------------------------------------------------------------------------

# 22. Phase 8 --- Validation Strategy

Do **not** use a naive:

``` python
train_test_split(all_pairs)
```

The evaluation unit is an **S1 entity**, not an independently sampled
pair.

The correct structure is:

``` text
All S1 entities
      ↓
S1-level development split
      ↓
Development S1
      ↓
candidate generation
      ↓
training pairs
      ↓
matcher

Validation S1
      ↓
complete blocking pipeline
      ↓
candidate pairs
      ↓
feature generation
      ↓
model scoring
      ↓
threshold/decision logic
      ↓
predicted ID sets
      ↓
entity-level F0.5
```

The key rule is:

> Keep the validation split at the S1-entity level.

This prevents candidate pairs belonging to the same S1 entity from
leaking between training and validation.

------------------------------------------------------------------------

# 23. Phase 9 --- F0.5 Metric

The competition uses **macro-averaged per-S1-entity F0.5**.

The metric is precision-heavy.

For every S1 entity:

``` text
TRUE = actual matched IDs
PRED = predicted matched IDs
```

F0.5 is calculated for that entity.

The per-entity results are then macro-averaged.

A particularly important edge case is:

``` text
TRUE = empty
PRED = empty
```

which receives:

``` text
F0.5 = 1.0
```

while predicting a match for an actually empty entity is penalized.

This is why singleton/no-match handling is important.

------------------------------------------------------------------------

# 24. Phase 10 --- Threshold Calibration

Do not automatically use:

``` text
probability > 0.50
```

The starter pipeline searches a threshold grid:

``` text
0.30 → 0.98
```

with:

``` text
0.02
```

steps.

The winning threshold should be selected using the **entity-level
validation F0.5**, not ordinary pairwise accuracy.

Once the real validation curve is available, the grid can be narrowed
around the best region for finer calibration.

------------------------------------------------------------------------

# 25. Phase 11 --- Singleton / Empty-Match Handling

The ground truth contains approximately:

``` text
123,247 / 2,206,821
≈ 5.59%
```

S1 entities with no matches.

Therefore, the system must explicitly handle:

``` text
No confident candidate
        ↓
predict empty match list
```

rather than forcing every S1 entity to receive a match.

False positive matches can be especially damaging because the metric is
precision-heavy.

------------------------------------------------------------------------

# 26. Phase 12 --- Final Test Inference

After validation is complete:

``` text
1. Build final indexes
2. Generate test candidates
3. Generate test features
4. Train/use final matcher
5. Apply calibrated threshold
6. Produce final S1 → matched IDs
```

The test pipeline must handle:

``` text
US
India
France
```

without assuming that France appeared in training.

------------------------------------------------------------------------

# 27. Final Output Files

The pipeline must generate:

``` text
output/
├── matching_results.tsv
└── candidate_pairs.tsv
```

## `matching_results.tsv`

Required columns:

``` text
source1_entity_id
matched_entity_ids
```

Every test S1 entity must appear exactly once.

If an S1 entity has no match:

``` text
matched_entity_ids = empty
```

Example:

``` text
source1_entity_id    matched_entity_ids
S1-001               S2-123,S3-456
S1-002
S1-003               S3-789
```

## `candidate_pairs.tsv`

This contains the final candidate pairs used by the matching model.

Every predicted match must exist in this file.

------------------------------------------------------------------------

# 28. Validate Every Submission

Before submitting to the leaderboard, run the organizer's validator.

From the project root:

``` powershell
python .\6ab10eb3b23ba_student_resource\student_resource\utils\validate_submission.py --matching .\output\matching_results.tsv --candidate .\output\candidate_pairs.tsv --test-dir .\6ab10eb3b23ba_student_resource\student_resource\dataset\test
```

Use the **official Amazon validator** whenever available.

The local validation pipeline is not a substitute for the organizer's
validator.

------------------------------------------------------------------------

# 29. Recommended Pipeline Commands

The master file supports stage-based execution.

From the project root:

``` powershell
python src/pipeline.py --stage audit
```

Run normalization:

``` powershell
python src/pipeline.py --stage normalize
```

Build blocking/index structures:

``` powershell
python src/pipeline.py --stage block
```

Train:

``` powershell
python src/pipeline.py --stage train
```

Validate:

``` powershell
python src/pipeline.py --stage validate
```

Prepare test data:

``` powershell
python src/pipeline.py --stage prepare-test
```

Run final prediction:

``` powershell
python src/pipeline.py --stage predict
```

Run the complete pipeline:

``` powershell
python src/pipeline.py --stage all
```

If your current `pipeline.py` exposes only a subset of these stages, use
the stages actually implemented by that version.

------------------------------------------------------------------------

# 30. Why Artifacts Exist

The dataset is too large to rebuild everything after every failure.

Instead:

``` text
pipeline
   ↓
artifacts/normalized
   ↓
artifacts/indexes
   ↓
artifacts/candidates
   ↓
artifacts/features
   ↓
artifacts/models
   ↓
output
```

Each phase should check whether its required artifact already exists.

Conceptually:

``` python
if artifact_exists():
    load_artifact()
else:
    create_artifact()
```

Therefore, if feature generation fails:

``` text
normalized     ✓
indexes        ✓
candidates     ✓
features       ✗
model          -
```

the pipeline can resume from features instead of rebuilding the entire
dataset.

------------------------------------------------------------------------

# 31. Important DuckDB / Parallel-Run Rule

If the pipeline uses a persistent DuckDB database or another file-backed
artifact, do **not** run two pipeline stages simultaneously against the
same database/artifact.

For example, do not run:

``` text
validation
```

and:

``` text
predict
```

at the same time if both access the same DuckDB file.

A lock error generally means another process currently owns the
database.

Before starting another stage:

1.  Check whether the previous process is still running.
2.  Let it finish, or stop it safely.
3.  Confirm that the database is no longer locked.
4.  Then start the next stage.

------------------------------------------------------------------------

# 32. GitHub --- What Should and Should Not Be Committed

The raw dataset is very large and should not be committed to GitHub.

Do not commit:

``` text
dataset/
*.zip
*.z01
*.z02
*.z03
__MACOSX/
.DS_Store
artifacts/
output/
models/
checkpoints/
```

The repository should contain:

``` text
src/
README.md
requirements.txt
.gitignore
documentation/code
```

and other lightweight project files required by the competition.

------------------------------------------------------------------------

# 33. Git Large-File Warning

If Git reports something like:

``` text
Writing objects: 100%
2.42 GiB
error: RPC failed; HTTP 408
```

stop pushing.

This usually means a large dataset/archive has accidentally entered Git
history.

Do not repeatedly run:

``` powershell
git push --force
```

until the large object has been removed from Git history.

Check:

``` powershell
git ls-files | Select-String "ezyZip|dataset|__MACOSX"
```

and:

``` powershell
git log --all --oneline -- ezyZip.z02
```

and:

``` powershell
git rev-list --objects --all | Select-String "ezyZip.z02"
```

Do not delete the local dataset merely to remove it from Git.

------------------------------------------------------------------------

# 34. Recommended `.gitignore`

Use:

``` gitignore
# Python
__pycache__/
*.py[cod]
.venv/
venv/
env/

# Jupyter
.ipynb_checkpoints/

# OS
.DS_Store
.DS_Store?
._*
Thumbs.db

# macOS extraction artifacts
__MACOSX/

# Archives
*.zip
*.z01
*.z02
*.z03
*.z04
*.z05
*.rar
*.7z

# Competition raw data
dataset/
**/dataset/

# Generated outputs
output/

# Intermediate artifacts
artifacts/
logs/
models/
checkpoints/

# IDE
.vscode/

# Temporary files
*.tmp
*.temp

# Python tooling
.pytest_cache/
.mypy_cache/
```

------------------------------------------------------------------------

# 35. Competition Submission Workflow

Use this order:

``` text
1. Download dataset
        ↓
2. Extract dataset
        ↓
3. Verify folder structure
        ↓
4. Audit dimensions
        ↓
5. Audit ground truth
        ↓
6. Audit countries/missing values
        ↓
7. Run normalization
        ↓
8. Build indexes
        ↓
9. Generate blocking candidates
        ↓
10. Measure candidate recall
        ↓
11. Build pair features
        ↓
12. Train matcher
        ↓
13. Tune entity-level F0.5 threshold
        ↓
14. Run validation
        ↓
15. Prepare test candidates
        ↓
16. Run test prediction
        ↓
17. Generate candidate_pairs.tsv
        ↓
18. Generate matching_results.tsv
        ↓
19. Run official validator
        ↓
20. Package final submission
```

------------------------------------------------------------------------

# 36. Do Not Burn Leaderboard Submissions

The competition guidelines specify a limited number of leaderboard
submissions.

According to the current project notes:

``` text
Maximum:
5 submissions/day
for 3 days
= 15 submissions
```

Therefore, do not submit every experimental run.

Before a leaderboard submission, record:

``` text
Date
Time
Experiment/version
Validation F0.5
Candidate recall
Threshold
Public leaderboard score
Notes
```

Maintain a submission history so that experiments remain reproducible.

------------------------------------------------------------------------

# 37. Experiment Log

Recommended format:

``` text
artifacts/
└── validation/
    ├── run_YYYYMMDD_HHMM/
    │   ├── metrics.json
    │   ├── threshold.json
    │   └── config.json
```

Also maintain a simple table:

  -----------------------------------------------------------------------------------
  Date         Run          Candidate    Val F0.5   Threshold    LB Score Change
                               Recall                                     
  ------------ ---------- ----------- ----------- ----------- ----------- -----------
  YYYY-MM-DD   baseline           ---         ---         ---         --- Initial

  YYYY-MM-DD   exp-02             ---         ---         ---         --- Hard
                                                                          negatives

  YYYY-MM-DD   exp-03             ---         ---         ---         --- Blocking
                                                                          change
  -----------------------------------------------------------------------------------

Do not change multiple major components at once when trying to
understand why validation changed.

------------------------------------------------------------------------

# 38. Current Known Gaps

These are the main areas to improve in priority order.

## 1. Country blocking

Verify whether country should be a mandatory restriction or merely a
blocking signal.

## 2. Hard-negative mining

Replace the initial "every candidate is a negative" strategy with
targeted hard negatives.

## 3. Blocking stopwords

Inspect high-frequency tokens and control candidate explosion.

## 4. Threshold resolution

Use the coarse threshold grid first, then search more finely around the
winning region.

## 5. Model constraints

XGBoost is the current structured-data model.

If neural/embedding models are added, verify:

-   license
-   parameter limit
-   competition rules

## 6. Rule-based overrides

A rule layer can be added after the ML threshold is tuned.

Only retain rules that demonstrate improvement on leakage-safe
validation.

------------------------------------------------------------------------

# 39. Development Philosophy

The project should optimize in this order:

``` text
Correctness
   ↓
Candidate Recall
   ↓
Candidate Volume
   ↓
Feature Quality
   ↓
Matcher Quality
   ↓
Threshold Calibration
   ↓
Runtime / Memory
   ↓
Leaderboard submission
```

Do not optimize the classifier before ensuring that blocking retains the
true matches.

A perfect classifier cannot recover a true match that was never included
in the candidate set.

------------------------------------------------------------------------

# 40. Final Competition Package

The final package should contain the required competition artifacts and
runnable code.

According to the current project notes, the final ZIP should contain:

``` text
output/
├── matching_results.tsv
└── candidate_pairs.tsv

code/
└── business_entity_resolution/
    └── runnable project code

Documentation_template.md
```

Before packaging:

1.  Run the official validator.
2.  Confirm both TSV files exist.
3.  Confirm the number of S1 predictions is correct.
4.  Confirm every predicted match appears in `candidate_pairs.tsv`.
5.  Confirm there are no duplicate S1 prediction rows.
6.  Confirm empty-match entities are represented correctly.
7.  Fill in `Documentation_template.md`.
8.  Package only the required files.

Do not include the raw multi-GB dataset unless the official submission
instructions explicitly require it.

------------------------------------------------------------------------

# 41. Hardware / Execution Constraints

The current competition notes specify:

-   Desktop/laptop execution
-   Single simultaneous login

Plan the pipeline accordingly.

Because the dataset is very large:

-   avoid unnecessary copies
-   avoid loading the same 5M-row file repeatedly
-   cache expensive artifacts
-   monitor RAM
-   monitor CPU
-   avoid simultaneous processes using the same database/artifact
-   run expensive stages sequentially

------------------------------------------------------------------------

# 42. Quick Start --- Complete Process

If starting from a clean machine:

### Step 1 --- Clone repository

``` powershell
git clone <YOUR_REPOSITORY_URL>
cd "amazon-ml-entity-resolution-2026"
```

### Step 2 --- Create environment

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3 --- Install dependencies

``` powershell
pip install -r requirements.txt
```

### Step 4 --- Download the official dataset

Use the Google Drive link at the top of this README.

### Step 5 --- Extract

Place the dataset under:

``` text
6ab10eb3b23ba_student_resource/
└── student_resource/
    └── dataset/
```

### Step 6 --- Verify files

``` powershell
dir .\6ab10eb3b23ba_student_resource\student_resource\dataset\train
dir .\6ab10eb3b23ba_student_resource\student_resource\dataset\test
```

### Step 7 --- Audit

Run the dimension and ground-truth audit commands in this README.

### Step 8 --- Run pipeline

``` powershell
python src/pipeline.py --stage audit
python src/pipeline.py --stage normalize
python src/pipeline.py --stage block
python src/pipeline.py --stage train
python src/pipeline.py --stage validate
```

Only after validation is complete:

``` powershell
python src/pipeline.py --stage prepare-test
python src/pipeline.py --stage predict
```

### Step 9 --- Validate output

``` powershell
python .\6ab10eb3b23ba_student_resource\student_resource\utils\validate_submission.py --matching .\output\matching_results.tsv --candidate .\output\candidate_pairs.tsv --test-dir .\6ab10eb3b23ba_student_resource\student_resource\dataset\test
```

### Step 10 --- Package

Prepare the final competition ZIP according to the official submission
instructions.

------------------------------------------------------------------------

# 43. Troubleshooting

## `FileNotFoundError`

Check your current directory:

``` powershell
pwd
```

If you are already inside:

``` text
...\dataset
```

do not use:

``` text
dataset/train/...
```

Use:

``` text
train/...
```

instead.

------------------------------------------------------------------------

## `origin already exists`

Check:

``` powershell
git remote -v
```

If the URL is correct, do nothing.

If it is incorrect:

``` powershell
git remote set-url origin <CORRECT_REPOSITORY_URL>
```

------------------------------------------------------------------------

## Git push fails with a multi-GB upload

Stop pushing.

Check for large files:

``` powershell
git ls-files | Select-String "ezyZip|dataset|__MACOSX"
```

If the large file has entered Git history, clean the history before
attempting another push.

------------------------------------------------------------------------

## DuckDB database is locked

Check whether another pipeline process is still running.

Do not start another stage against the same database simultaneously.

Wait for the current stage to finish or stop it safely, then retry.

------------------------------------------------------------------------

## Candidate recall is low

Do not immediately change the XGBoost model.

Investigate:

``` text
normalization
blocking rules
stopwords
country restriction
name blocks
address blocks
postal blocks
exact-match blocks
```

Candidate recall is a ceiling on downstream matching performance.

------------------------------------------------------------------------

## Candidate count is extremely high

Investigate:

``` text
high-frequency name tokens
high-frequency address tokens
common city names
generic business terms
country intersections
```

Use frequency statistics to refine blocking.

------------------------------------------------------------------------

# 44. Final Mental Model

Think of the competition as:

``` text
12.5M training records
        ↓
NORMALIZE
        ↓
BLOCK
        ↓
millions → manageable candidate pairs
        ↓
PAIR FEATURES
        ↓
XGBoost MATCHER
        ↓
ENTITY-LEVEL THRESHOLD
        ↓
S1 → {S2/S3 IDs}
        ↓
F0.5
```

The two most important principles are:

### Principle 1

**Blocking determines the maximum achievable recall.**

### Principle 2

**The final metric is evaluated at the S1-entity level, so validation
must also be entity-level.**

------------------------------------------------------------------------

## Official Deliverables

Before submitting, the project should produce:

``` text
output/
├── matching_results.tsv
└── candidate_pairs.tsv
```

and the final competition package should contain the required runnable
code and completed documentation.

**Dataset Drive:** `[PASTE OFFICIAL DATASET DRIVE LINK HERE]`
