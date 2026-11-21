# MP203: PG Room Rent (Kochi)

SCSE3040 MLOps mini project. **Group G219**, problem **MP203**. Batch 12, faculty Dr. Upasna Joshi.

| Enrolment no. | Name | Role |
|---|---|---|
| S24CSEU1080 | Pradhumn Koshta | Model lead |
| S24CSEU1069 | Arushi Singh | Data lead |
| S24CSEU1052 | Anshul Mittal | Ops lead |
| S24CSEU1061 | Adarsh Raj | Service lead |

## The problem

The Kochi Student Housing Board wants a **fair monthly rent** (`rent_rs`, in Rs per month) for every
PG room listed, so that students can spot an over-priced room before they sign. This is a regression
problem, measured with **MAE** (the average mistake, in rupees per month).

## Results

| What | Value |
|---|---|
| v1 (Ridge, alpha 1) validation MAE, last 20% of train by date | 785 Rs/month |
| Most drifted feature (PSI, train vs later) | `room_area_sqft` (PSI 6.63) |
| Worst-served locality for v1 on later.csv | Sector West (MAE 7,598) |
| v1 vs v2 on the last 30% of later.csv | 5,242 vs 1,051 Rs/month |
| Decision (promote if v2 is at least 15% better) | **promote**: v2 is `@champion` |

Data problems found in `train.csv` (all found by `src/pgrent/data_checks.py`):

| Problem | Present | Details |
|---|---|---|
| Leaking column | yes | `rent_in_signed_agreement` (filled after the deal; missing from holdout) |
| Unit mix-up | no | every locality's medians agree |
| Sentinel value | no | no special number repeats outside the clean range |
| Duplicate rows | no | 0 extra copies |
| Stuck sensor | yes | `distance_to_campus_km` = 1.27 from 2026-05-06 to 2026-05-25 (104 rows) |
| Target typos | yes | 14 rents typed 10x too big |
| Spelling variants | yes | `furnishing`: 18 spellings, 3 real levels |

All numbers are also in `reports/findings.yaml` and `reports/monitoring.json`.

## How to run

Run everything from the project root. The CSVs are tracked with DVC, so copy them from the problem
zip into `data/raw/` first (`train.csv`, `later.csv`, `holdout_features.csv`).

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
export PYTHONPATH=src

.venv/bin/python -m pgrent.data_checks   # find the 7 data problems
.venv/bin/python -m pgrent.clean         # fix them -> data/processed/train_clean.csv
.venv/bin/python -m pgrent.train         # 4 candidate runs + v1, registered as @champion
.venv/bin/python -m pgrent.monitor       # PSI, MAE per locality, v2, promotion rule
.venv/bin/python -m pgrent.predict       # @champion -> predictions/holdout_predictions.csv

.venv/bin/pytest                          # tests (no CSVs needed)
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db   # look at the runs
```

`train.py` resets the model registry, so run `train` and then `monitor` once each, in that order.

### API, Docker and Kubernetes

```bash
uvicorn app.main:app --port 8000                 # local
docker build -t mp203-api:1.0 .
docker run -p 8000:8000 mp203-api:1.0

curl localhost:8000/health
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{
  "distance_to_campus_km": 4.5, "room_area_sqft": 350, "sharing": 2, "meals_per_day": 2,
  "building_age_yr": 15, "wifi_mbps": 100, "furnishing": "semi", "locality": "Sector West"}'

kubectl apply -f k8s/          # Deployment mp203-api (4 replicas) + Service mp203-svc (80 -> 8000)
```

The client's two rules are part of the request schema, so a bad request gets **422**:
`room_area_sqft` above 650 is rejected, and `wifi_mbps` below 0 is rejected.

## Layout

```
params.yaml             every setting (paths, columns, check thresholds, models, MLflow names)
src/pgrent/
  config.py             reads params.yaml, SHA-256 of train.csv
  data_checks.py        the seven data checks
  clean.py              the fixes (uses the checks' results)
  train.py              v1 with MLflow tracking + registry
  monitor.py            PSI, MAE per group, v2 retrain, promotion rule
  predict.py            holdout predictions from @champion
app/main.py             FastAPI service (/predict, /health)
app/json_logging.py     one JSON log line per event, to stdout
tests/                  pytest (TestClient for the API)
data/raw/*.dvc          DVC pointers to train.csv and later.csv
mlflow.db               MLflow tracking database (runs + model registry)
models/model.joblib     the champion model the API loads
Dockerfile, .dockerignore, .github/workflows/ci.yml, k8s/
reports/findings.yaml   the filled findings form
predictions/holdout_predictions.csv
```

## Design notes

- **Leak.** `rent_in_signed_agreement` follows the target almost exactly (Spearman 0.98), but it is
  written down only after the rent is agreed, and the holdout file does not have it. It is dropped.
- **Cleaning.** Stuck readings become blanks, and the pipeline fills them with the median. Typo
  rents are divided by 10. Spellings are mapped to the clean levels in `later.csv`.
- **Validation is by date**, never random, because the model will always predict the future.
- **Why v2 weights later rows.** Sector West rents jumped in July: v1 under-predicts them by about
  7,600 Rs every month. With 1,000 old rows and only 210 new ones, a plain retrain moves only part
  of the way. v2 therefore counts each later row more. The weight (1, 5, 20 or 100) is picked using
  only the first 70% of later.csv; the last 30% stays untouched for the v1 vs v2 comparison.
