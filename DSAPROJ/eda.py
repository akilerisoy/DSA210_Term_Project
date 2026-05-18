# DSA 210 EDA 

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("data/merged_dataset.csv")
df["market_value_eur"] = pd.to_numeric(df["market_value_eur"], errors="coerce")
os.makedirs("figures", exist_ok=True)

print(" ")
print(f"Dataset: {len(df)} players, {df.shape[1]} columns\n")


# Simplify positions into 4 groups
def simplify_position(pos):
    pos = str(pos).split(",")[0].strip().upper()
    if pos == "FW":   return "Forward"
    elif pos == "MF": return "Midfielder"
    elif pos == "DF": return "Defender"
    elif pos == "GK": return "Goalkeeper"
    return "Other"

df["position_group"] = df["Pos"].apply(simplify_position)
df = df[df["position_group"] != "Other"]
df["league_short"] = df["Comp"].str.split(" ", n=1).str[-1]


# 1. Descriptive statistics
print("Market Value Stats")
print(f"  Mean   : €{df['market_value_eur'].mean()/1e6:.1f}M")
print(f"  Median : €{df['market_value_eur'].median()/1e6:.1f}M")
print(f"  Min    : €{df['market_value_eur'].min()/1e6:.1f}M")
print(f"  Max    : €{df['market_value_eur'].max()/1e6:.1f}M\n")

print("Players per Position")
for pos, count in df["position_group"].value_counts().items():
    print(f"  {pos}: {count}")

print("\nPlayers per League")
for league, count in df["league_short"].value_counts().items():
    print(f"  {league}: {count}")
print()


# 2. Market value distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df["market_value_eur"] / 1e6, bins=50, color="#2E75B6", edgecolor="white")
axes[0].set_title("Market Value Distribution", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Market Value (€ millions)")
axes[0].set_ylabel("Number of Players")

axes[1].hist(df["log_market_value"], bins=50, color="#70AD47", edgecolor="white")
axes[1].set_title("Log Market Value Distribution", fontsize=13, fontweight="bold")
axes[1].set_xlabel("log(Market Value)")
axes[1].set_ylabel("Number of Players")

plt.tight_layout()
plt.savefig("figures/01_market_value_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 01_market_value_distribution.png")


# 3. Market value by position
position_order = ["Forward", "Midfielder", "Defender", "Goalkeeper"]
colors = ["#C00000", "#2E75B6", "#70AD47", "#FFC000"]
medians = [df[df["position_group"] == p]["market_value_eur"].median() / 1e6 for p in position_order]

fig, ax = plt.subplots(figsize=(11, 8))
bars = ax.bar(position_order, medians, color=colors, edgecolor="white", width=0.6)
for bar, val in zip(bars, medians):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.2,
            f"€{val:.1f}M", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title("Median Market Value by Position", fontsize=13, fontweight="bold")
ax.set_ylabel("Median Market Value (€ millions)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:.0f}M"))
plt.tight_layout()
plt.savefig("figures/02_market_value_by_position.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 02_market_value_by_position.png")


# 4. Market value by league
league_medians = (df.groupby("league_short")["market_value_eur"]
                    .median().sort_values(ascending=False))

fig, ax = plt.subplots(figsize=(11, 8))
bars = ax.bar(league_medians.index, league_medians.values / 1e6,
              color="#2E75B6", edgecolor="white", width=0.6)
for bar, val in zip(bars, league_medians.values / 1e6):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.2,
            f"€{val:.1f}M", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title("Median Market Value by League", fontsize=13, fontweight="bold")
ax.set_ylabel("Median Market Value (€ millions)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:.0f}M"))
plt.tight_layout()
plt.savefig("figures/03_market_value_by_league.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 03_market_value_by_league.png")


# 5. Age vs market value
fig, ax = plt.subplots(figsize=(11, 8))
ax.scatter(df["age"], df["market_value_eur"] / 1e6, alpha=0.6, color="#2E75B6", s=15)
ax.set_title("Age vs Market Value", fontsize=13, fontweight="bold")
ax.set_xlabel("Age")
ax.set_ylabel("Market Value (€ millions)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:.0f}M"))
plt.tight_layout()
plt.savefig("figures/04_age_vs_market_value.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 04_age_vs_market_value.png")


# 6. Top 10 features correlated with log market value
numeric_cols = df.select_dtypes(include="number").columns.tolist()
exclude = ["market_value_eur", "log_market_value", "player_id", "age"]
feature_cols = [c for c in numeric_cols if c not in exclude]

correlations = (df[feature_cols]
                .corrwith(df["log_market_value"])
                .abs()
                .sort_values(ascending=False)
                .head(10))

fig, ax = plt.subplots(figsize=(11, 8))
bars = ax.barh(correlations.index[::-1], correlations.values[::-1],
               color="#2E75B6", edgecolor="white")
for bar, val in zip(bars, correlations.values[::-1]):
    ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9)
ax.set_title("Top 10 Features Correlated with Market Value", fontsize=13, fontweight="bold")
ax.set_xlabel("Absolute Correlation with log(Market Value)")
plt.tight_layout()
plt.savefig("figures/05_top_correlations.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 05_top_correlations.png")


# Summary
print("\nSUMMARY")
print(f"  Total players  : {len(df)}")
print(f"  Median value   : €{df['market_value_eur'].median()/1e6:.1f}M")
print(f"  Mean value     : €{df['market_value_eur'].mean()/1e6:.1f}M")
print(f"  Top correlated : {correlations.index[0]} (r = {correlations.values[0]:.3f})")
print("\nAll figures saved to figures/")
print(" ")