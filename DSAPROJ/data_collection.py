# DSA 210 Data Collection        

import pandas as pd
import math
import os
import warnings
warnings.filterwarnings("ignore")

os.makedirs("data", exist_ok=True)


# 1. Load FBref CSV
fbref_df = pd.read_csv("fbref_standard.csv")
fbref_df = fbref_df[fbref_df["Player"].notna()]
fbref_df = fbref_df[fbref_df["Player"] != "Player"]

numeric_cols = ["Min", "MP", "Starts", "Gls", "Ast", "xG", "xAG",
                "npxG", "PrgC", "PrgP", "PrgR", "CrdY", "CrdR"]
for col in numeric_cols:
    if col in fbref_df.columns:
        fbref_df[col] = pd.to_numeric(fbref_df[col], errors="coerce")


print(" ")
print(f"FBref loaded: {len(fbref_df)} players")


# 2. Load Transfermarkt players.csv
tm_df = pd.read_csv("players.csv")
tm_df = tm_df[["name", "current_club_name", "position",
               "date_of_birth", "market_value_in_eur"]].copy()
tm_df = tm_df.rename(columns={
    "name": "player_tm",
    "current_club_name": "club_tm",
    "market_value_in_eur": "market_value_eur"
})
tm_df = tm_df.dropna(subset=["market_value_eur"])
print(f"Transfermarkt loaded: {len(tm_df)} players")


# 3. Merge on normalized player name
fbref_df["player_norm"] = fbref_df["Player"].str.lower().str.strip()
tm_df["player_norm"]    = tm_df["player_tm"].str.lower().str.strip()

merged = fbref_df.merge(tm_df, on="player_norm", how="inner")
print(f"Matched: {len(merged)} players")


# 4. Filter and clean
merged = merged[merged["Min"] >= 500].copy()
merged = (merged.sort_values("Min", ascending=False)
                .drop_duplicates(subset=["Player"], keep="first"))

merged["log_market_value"] = merged["market_value_eur"].apply(lambda x: math.log1p(x))
merged["league_short"] = merged["Comp"].str.split(" ", n=1).str[-1]
merged["date_of_birth"] = pd.to_datetime(merged["date_of_birth"], errors="coerce")
merged["age"] = ((pd.Timestamp("2025-01-01") - merged["date_of_birth"]).dt.days / 365.25).round(1)

merged.to_csv("data/merged_dataset.csv", index=False)
print(f"Final dataset: {len(merged)} players, {merged.shape[1]} columns")
print("Saved to data/merged_dataset.csv")
print(" ")