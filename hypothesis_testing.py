# DSA 210 Hypothesis Testing

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import kruskal, mannwhitneyu, shapiro, spearmanr
import os
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("data/merged_dataset.csv")
df["market_value_eur"] = pd.to_numeric(df["market_value_eur"], errors="coerce")
df["Min"] = pd.to_numeric(df["Min"], errors="coerce")
os.makedirs("figures", exist_ok=True)

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

ALPHA = 0.05

print("DSA 210 Hypothesis Testing")
print(" ")
print(f"Dataset: {len(df)} players\n")


# Normality Check 
sample = df["market_value_eur"].dropna().sample(min(500, len(df)), random_state=42)
stat, p = shapiro(sample)
normal = p >= ALPHA
print("Normality Check")
print(f"  Market values are NOT normally distributed.")
print(f"  Using non-parametric tests.\n")


# Hypothesis 1: Market value across positions 
positions = ["Forward", "Midfielder", "Defender", "Goalkeeper"]
groups = [df[df["position_group"] == p]["market_value_eur"].dropna() for p in positions]
stat, p_val = kruskal(*groups)

print("Hypothesis 1: Do market values differ across positions?")
for p in positions:
    median = df[df["position_group"] == p]["market_value_eur"].median() / 1e6
    print(f"  {p}: median = €{median:.1f}M")
result1 = "Yes" if p_val < ALPHA else "No"
print(f"  Result: {result1}, market values differ across positions.")

# Pairwise
n_comparisons = len(positions) * (len(positions) - 1) // 2
bonferroni_alpha = ALPHA / n_comparisons
print("  Pairwise:")
for i in range(len(positions)):
    for j in range(i + 1, len(positions)):
        g1 = df[df["position_group"] == positions[i]]["market_value_eur"].dropna()
        g2 = df[df["position_group"] == positions[j]]["market_value_eur"].dropna()
        _, p_pair = mannwhitneyu(g1, g2, alternative="two-sided")
        sig = "Significant" if p_pair < bonferroni_alpha else "Not significant"
        print(f"  {positions[i]} vs {positions[j]}: {sig}")

# Plot
medians = [df[df["position_group"] == p]["market_value_eur"].median() / 1e6 for p in positions]
colors = ["#C00000", "#2E75B6", "#70AD47", "#FFC000"]
fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.bar(positions, medians, color=colors, edgecolor="white", width=0.6)
for bar, val in zip(bars, medians):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.2,
            f"€{val:.1f}M", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title(f"Median Market Value by Position\n(Kruskal-Wallis p = {p_val:.2e})", fontsize=13, fontweight="bold")
ax.set_ylabel("Median Market Value (€ millions)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:.0f}M"))
plt.tight_layout()
plt.savefig("figures/06_hypothesis1_positions.png", dpi=150, bbox_inches="tight")
plt.close()
print("")


# Hypothesis 2: Market value across leagues 
leagues = df["league_short"].unique().tolist()
league_groups = [df[df["league_short"] == l]["market_value_eur"].dropna() for l in leagues]
stat, p_val2 = kruskal(*league_groups)

print("Hypothesis 2: Do market values differ across leagues?")
league_order = (df.groupby("league_short")["market_value_eur"]
                .median().sort_values(ascending=False).index.tolist())
for l in league_order:
    median = df[df["league_short"] == l]["market_value_eur"].median() / 1e6
    print(f"  {l}: median = €{median:.1f}M")
result2 = "Yes" if p_val2 < ALPHA else "No"
print(f"  Result: {result2}, market values differ across leagues.")

# Plot
league_medians = [df[df["league_short"] == l]["market_value_eur"].median() / 1e6 for l in league_order]
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(league_order, league_medians, color="#2E75B6", edgecolor="white", width=0.6)
for bar, val in zip(bars, league_medians):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.2,
            f"€{val:.1f}M", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title(f"Median Market Value by League\n(Kruskal-Wallis p = {p_val2:.2e})", fontsize=13, fontweight="bold")
ax.set_ylabel("Median Market Value (€ millions)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:.0f}M"))
plt.tight_layout()
plt.savefig("figures/07_hypothesis2_leagues.png", dpi=150, bbox_inches="tight")
plt.close()
print("")


# Hypothesis 3: Forwards vs Defenders 
forwards  = df[df["position_group"] == "Forward"]["market_value_eur"].dropna()
defenders = df[df["position_group"] == "Defender"]["market_value_eur"].dropna()
_, p_val3 = mannwhitneyu(forwards, defenders, alternative="greater")

print("Hypothesis 3: Are Forwards worth more than Defenders?")
print(f"  Forwards median  = €{forwards.median()/1e6:.1f}M")
print(f"  Defenders median = €{defenders.median()/1e6:.1f}M")
result3 = "Yes" if p_val3 < ALPHA else "No"
print(f"  Result: {result3}, Forwards are significantly more valuable.")

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.boxplot([forwards / 1e6, defenders / 1e6], labels=["Forwards", "Defenders"],
           patch_artist=True,
           boxprops=dict(facecolor="#2E75B6", color="white"),
           medianprops=dict(color="white", linewidth=2),
           showfliers=False)
ax.set_title(f"Forwards vs Defenders Market Value\n(Mann-Whitney p = {p_val3:.2e})", fontsize=13, fontweight="bold")
ax.set_ylabel("Market Value (€ millions)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:.0f}M"))
plt.tight_layout()
plt.savefig("figures/08_hypothesis3_fwd_vs_def.png", dpi=150, bbox_inches="tight")
plt.close()
print("")


# Hypothesis 4: xG vs Market Value 
xg_col = [c for c in df.columns if "xg" in c.lower() and "xa" not in c.lower()
          and "per" not in c.lower() and "90" not in c.lower()]

if xg_col:
    xg_col = xg_col[0]
    valid = df[[xg_col, "market_value_eur"]].dropna()
    rho, p_val4 = spearmanr(valid[xg_col], valid["market_value_eur"])

    print("Hypothesis 4: Does xG correlate with market value?")
    print(f"  Correlation = {rho:.3f} (range: -1 to 1, higher = stronger)")
    result4 = "Yes" if p_val4 < ALPHA and rho > 0 else "No"
    print(f"  Result: {result4}, xG is significantly correlated with market value.")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(valid[xg_col], valid["market_value_eur"] / 1e6, alpha=0.3, color="#2E75B6", s=15)
    ax.set_title(f"xG vs Market Value\n(Spearman ρ = {rho:.3f}, p = {p_val4:.2e})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Expected Goals (xG)")
    ax.set_ylabel("Market Value (€ millions)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:.0f}M"))
    plt.tight_layout()
    plt.savefig("figures/09_hypothesis4_xg_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("")


# Summary 
print("SUMMARY")
print(f"  H1: Values differ across positions  :  {'Rejected' if p_val  < ALPHA else 'Not Rejected'}")
print(f"  H2: Values differ across leagues    :  {'Rejected' if p_val2 < ALPHA else 'Not Rejected'}")
print(f"  H3: Forwards worth more             :  {'Rejected' if p_val3 < ALPHA else 'Not Rejected'}")
if xg_col:
    print(f"  H4: xG correlates with value        :  {'Rejected' if p_val4 < ALPHA else 'Not Rejected'}")
print(" ")