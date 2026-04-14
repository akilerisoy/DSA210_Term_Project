# DSA 210 - Football Player Market Value Analysis

## Project Overview
This project investigates whether a football player's transfer market value can be predicted using seasonal performance statistics from Europe's top 5 leagues (2024-25 season).

## Data Sources
- **FBref** (fbref_standard.csv) — Player performance statistics from [Kaggle - orkunaktas/all-football-players-stats-in-top-5-leagues-2425](https://www.kaggle.com/datasets/orkunaktas/all-football-players-stats-in-top-5-leagues-2425), originally sourced from FBref.com
- **Transfermarkt** (players.csv) — Player market valuations from [Kaggle - davidcariboo/player-scores](https://www.kaggle.com/datasets/davidcariboo/player-scores)

## How to Run
Install dependencies:
```
pip install -r requirements.txt
```
Run scripts in order:
```
python data_collection.py
python eda.py
python hypothesis_testing.py
```

## Hypothesis Tests
- H1: Market value differs across positions
- H2: Market value differs across leagues
- H3: Forwards are worth more than Defenders
- H4: xG is positively correlated with market value

## Files
- `data_collection.py` — loads and merges the datasets
- `eda.py` — exploratory data analysis and charts
- `hypothesis_testing.py` — statistical hypothesis tests
- `figures/` — all output charts
