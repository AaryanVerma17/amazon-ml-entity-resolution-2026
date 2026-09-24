# Amazon ML Challenge 2026 — Business Entity Resolution (starter pipeline)

Skeleton, smoke-tested end-to-end on synthetic data. Fill it in once the real
dataset ZIP is available.

## Layout
```
src/
  normalize.py      name/address normalization (keeps raw + expanded forms)
  blocking.py       union-blocking candidate generation (country/name-token/
                     addr-token/exact-name/postal)
  features.py       ~20 pairwise similarity features (Levenshtein, token
                     sort/set ratio, Jaccard, TF-IDF char n-gram cosine,
                     postal/house-number match, cross features)
  model.py          XGBoost classifier + threshold search that optimizes
                     macro-averaged per-entity F0.5 (the real metric)
  metrics.py         F0.5 exactly as scored: per-S1-entity, empty/empty = 1.0
  pipeline.py        orchestrates load -> block -> feature -> train ->
                     tune -> predict -> write both TSVs
  make_synthetic.py  generates fake data matching the schema, for smoke tests
dataset/             put the real train/ and test/ folders here
output/               matching_results.tsv + candidate_pairs.tsv land here
```

## Run
```bash
pip install -r requirements.txt
# once real dataset ZIP is extracted into dataset/train and dataset/test:
cd src && python pipeline.py --data-dir ../dataset --out-dir ../output

# validate before every leaderboard submission (use Amazon's real validator,
# not a substitute):
python3 utils/validate_submission.py \
  --matching ../output/matching_results.tsv \
  --candidate ../output/candidate_pairs.tsv \
  --test-dir ../dataset/test
```

## Known gaps to close once real data lands (in priority order)
1. **Country blocking assumption** — `blocking.py` currently intersects
   candidates with the same-country pool as a precision boost. Check this
   against real data first: if country labels are noisy/missing, drop that
   intersection (it's one `&` away from being pure union).
2. **Negative sampling** — currently trains on *every* generated candidate as
   a negative unless it's a true match. On the real (larger) dataset this
   will be extremely imbalanced; add explicit hard-negative mining (same
   name token, different address) per section 21 of the plan.
3. **Blocking token stopword list** — `STOPWORD_TOKENS` in `blocking.py` is a
   short manual list; extend it once you see real high-frequency tokens
   (e.g. common city names) inflating candidate volume.
4. **Threshold grid resolution** — currently 0.02 steps from 0.30–0.98;
   narrow around the winning region once you see the real validation curve.
5. **Model size/license constraint** — XGBoost (Apache-2.0) is fine as-is;
   if you add a neural/embedding component later, re-check it against the
   MIT/Apache-2.0 + ≤8B-parameter constraint in the guidelines.
6. **Rule-based override layer** (plan section 18) — not yet implemented.
   Consider adding after the ML threshold is tuned, only if validation shows
   specific high-confidence patterns the model underweights.

## Submission checklist (per the official guidelines)
- Max 5 leaderboard submissions/day for 3 days (15 total) — don't burn them
  before checking `[val] candidate recall ceiling` and the F0.5 table printed
  by `pipeline.py`.
- Keep a dated log of every submission's val F0.5 + public leaderboard score
  (guidelines require submission version history).
- Final zip needs: `output/` (both TSVs), `code/business_entity_resolution/`
  (this folder, runnable), and the filled-in `Documentation_template.md`.
- Desktop/laptop only, single simultaneous login.
