"""Build and execute notebooks/walkthrough.ipynb (run after `python -m src.train`).

    python notebooks/build_walkthrough.py
"""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("# Football transfer valuation: walkthrough\n\n"
       "Predicting a player's Transfermarkt market value at the time of a transfer from the "
       "previous season's FBref stats and the clubs involved. This notebook reads the outputs of "
       "`python -m src.train` and shows the data, the two silent bugs, the fixes and the results."),
    code("import json, sys\nfrom pathlib import Path\nimport numpy as np, pandas as pd\n"
         "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
         "sys.path.insert(0, str(ROOT))\n"
         "from src.config import DATASET, BUILD_STATS, TM_VALUATIONS, FBREF_STATS, RAW_FBREF_DIR\n"
         "pd.set_option('display.width', 120)"),

    md("## 1. The dataset\n\nOne row per transfer, joined to the player's FBref season before the move. "
       "Training seasons end 2019 to 2024; 2025 is held out."),
    code("df = pd.read_parquet(DATASET)\n"
         "print(df.shape)\n"
         "df.groupby('season_end_year').size().rename('transfers').to_frame().T"),
    code("stats = json.loads(BUILD_STATS.read_text())\n"
         "n = stats['transfers_in_window']\n"
         "pd.DataFrame({'transfers': stats, 'share of window': {k: f'{v / n:.2%}' for k, v in stats.items()}})"),

    md("## 2. Why the name join needed a birth year\n\n"
       "The FBref source keeps one row per name per season across all ten leagues, so namesakes are "
       "summed. 'Fernando' at Almeria in 2022-23 played 89 league matches, which is impossible."),
    code("raw = pd.read_csv(RAW_FBREF_DIR / 'fifa_fbref_merged.csv', low_memory=False,\n"
         "                  usecols=['player', 'team', 'league', 'season', 'Playing Time_MP', 'Playing Time_Min', 'born_fbref', 'long_name'])\n"
         "raw[(raw.player == 'Fernando') & (raw.season == 2223)]"),
    code("raw.groupby(['player', 'season'])['team'].nunique().max()  # never more than one row per name and season"),

    md("## 3. Silent bug 1: position flags were always zero\n\n"
       "The flags compared Transfermarkt's `position` with FBref codes (GK, DF, MF, FW). The column "
       "actually holds the labels below, so the original check matched nothing."),
    code("print(df['position'].value_counts())\n"
         "print('old check matches:', df['position'].str.upper().isin(['GK', 'DF', 'MF', 'FW']).sum())\n"
         "df[['is_goalkeeper', 'is_defender', 'is_midfielder', 'is_forward']].sum()"),

    md("## 4. Silent bug 2: the contract feature, and a third in the club features\n\n"
       "The contract feature first used a 2025 contract snapshot for transfers back to 2018; its "
       "replacement used the date of the player's *next* transfer (future information). It is removed. "
       "The club-level features had the same kind of problem: `current_club_id` in the valuations is "
       "the scrape-time club, identical across a player's whole history."),
    code("vals = pd.read_parquet(TM_VALUATIONS)\n"
         "per_player = vals.groupby('player_id')['current_club_id'].nunique()\n"
         "print('players with more than one club across all their valuations:', int((per_player > 1).sum()))"),

    md("## 5. Results"),
    code("from IPython.display import Markdown\n"
         "text = (ROOT / 'results' / 'metrics.md').read_text(encoding='utf-8')\n"
         "Markdown(text.split('## Test R squared by position')[0])"),
    code("Markdown((ROOT / 'results' / 'fix_log.md').read_text(encoding='utf-8').split('## Match rate')[0])"),
    code("from IPython.display import Image\nImage(str(ROOT / 'results' / 'feature_importance.png'), width=640)"),

    md("## 6. Where the model is wrong\n\nLargest 2025 errors for XGBoost with modern metrics, in euros."),
    code("import joblib\nfrom src.config import MODEL_DIR\n"
         "b = joblib.load(MODEL_DIR / 'xgboost_with_modern.joblib')\n"
         "test = df[df.season_end_year == 2025].copy()\n"
         "test['predicted_eur'] = np.expm1(b['model'].predict(b['preprocessor'].transform(test[b['num_cols'] + b['cat_cols']])))\n"
         "test['ratio'] = test['predicted_eur'] / test['market_value_eur_at_transfer']\n"
         "cols = ['player', 'transfer_date', 'league', 'age_at_transfer', 'market_value_eur_at_transfer', 'predicted_eur', 'ratio']\n"
         "test.loc[test['ratio'].sub(1).abs().sort_values(ascending=False).index[:8], cols].round(2)"),
    code("print('share of 2025 predictions within a factor of 2 of the real value:',\n"
         "      f\"{((test.ratio > 0.5) & (test.ratio < 2)).mean():.1%}\")"),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
nbf.write(nb, HERE / "walkthrough.ipynb")
print("wrote", HERE / "walkthrough.ipynb")
