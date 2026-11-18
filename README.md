# Player Market Value Prediction

End-to-end pipeline that predicts a footballer's Transfermarkt market value
from FBref season stats + Transfermarkt transfer/club context, with three
algorithms (Ridge / Random Forest / XGBoost) trained on two feature sets
(traditional only / traditional + modern).

## Layout

```
src/                 source modules
  transfermarkt.py   raw csv  -> tm_*.parquet                (intermediate)
  fbref.py           raw csv  -> fbref_player_season_stats.parquet
  build_dataset.py   intermediate -> data/final/player_market_value_dataset.parquet
  models_v2.py       trains 6 models, writes ./models/*.joblib
  club_lookup.py     fuzzy club name -> log1p(median MV) feature for predict
  predict.py         scores theoretical players using all 6 saved bundles
data/
  raw/               source CSVs (transfermarkt + fbref)
  intermediate/      parquet outputs of transfermarkt.py / fbref.py
  final/             output of build_dataset.py
models/              joblib bundles produced by models_v2.py
experiment.ipynb     interactive notebook for scoring theoretical players
requirements.txt     pip dependencies
```

## Setup (any machine)

```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Running the pipeline

The intermediate parquets, the final dataset, and the trained models are all
already included, so you can jump straight to step 4 if you only want to use
the model. To rebuild everything from raw CSVs:

```bash
python src/transfermarkt.py    # 1. raw -> tm_*.parquet
python src/fbref.py            # 2. raw -> fbref_player_season_stats.parquet
python src/build_dataset.py    # 3. intermediate -> final dataset
python src/models_v2.py        # 4. train + save 6 (preprocessor, model) bundles
```

Step 4 takes the longest (~10–20 min) because of `RandomizedSearchCV` tuning.

## Scoring a theoretical player

After step 4 has produced the bundles in `./models/`:

```bash
jupyter notebook experiment.ipynb
```

Edit the `player` dict in cell 3 and re-run that cell; the notebook prints
predicted euro values from all six (algorithm × feature-set) combinations.
Cell 6 sweeps a single input (default: age) so you can see how predictions
move along that axis.
