"""
trend_analysis.py - Temporal Sentiment Trend Analysis and Visualization.
Directly reproduces Table 3 and Figure 1 from Section 6 of the paper:
- Monthly aggregation of predicted positive sentiment ratio across Dec 31 2020 - Apr 2021
- Dual-axis time series plot comparing % Positive Posts against National COVID-19 Vaccine Doses Administered
Authors: Navaneeth Tanuboddi (PES1UG24CS709), Krrish P Raju (PES1UG24CS708)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, Any


def plot_sentiment_vs_vaccine_doses(timeline_csv: str, output_image_path: str):
    """
    Generates Figure 1 from Section 6 of the paper:
    Dual-axis line plot showing:
    - Left Y-Axis: Percentage of Positive Posts (%)
    - Right Y-Axis: COVID-19 vaccine doses administered (Millions)
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_image_path)), exist_ok=True)
    df = pd.read_csv(timeline_csv)
    
    fig, ax1 = plt.subplots(figsize=(10, 5.5), dpi=300)
    
    # Left axis: Sentiment percentage
    color1 = "#d95f02" # Orange-brown as in paper
    ax1.set_xlabel("Date (2020-12 to 2021-04)", fontsize=12, fontweight="bold", labelpad=10)
    ax1.set_ylabel("Percentage of Positive Posts (%)", color=color1, fontsize=12, fontweight="bold")
    line1 = ax1.plot(df["month"], df["sentiment_pct"], color=color1, linewidth=2.8, marker="o", label="Percentage of Positive Posts")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, 105)
    ax1.set_yticks(np.arange(0, 110, 10))
    ax1.grid(True, linestyle="--", alpha=0.4)
    
    # Right axis: Doses administered in millions
    ax2 = ax1.twinx()
    color2 = "#1f78b4" # Blue
    ax2.set_ylabel("Vaccine Doses Administered (Millions)", color=color2, fontsize=12, fontweight="bold")
    line2 = ax2.plot(df["month"], df["doses_millions"], color=color2, linewidth=2.8, marker="s", label="COVID-19 vaccine doses administered")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, 300)
    ax2.set_yticks(np.arange(0, 350, 50))
    
    # Title & Legend
    plt.title("COVID-19 Vaccine Doses & Sentiment Trend Over Time", fontsize=14, fontweight="bold", pad=15)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower left", framealpha=0.9)
    
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(output_image_path, bbox_inches="tight")
    plt.close()
    print(f"Figure 1 saved to: {output_image_path}")


def compute_monthly_summary(monthly_posts_csv: str, models: Dict[str, Any]) -> pd.DataFrame:
    """
    Computes predicted sentiment statistics grouped by month, matching Table 3 from the paper.
    """
    df = pd.read_csv(monthly_posts_csv)
    months = ["December 31", "January", "February", "March", "April"]
    
    results = []
    
    for m in months:
        subset = df[df["month"] == m]
        if subset.empty:
            continue
            
        texts = subset["text"].tolist()
        ground_truth = subset["ground_truth_label"].tolist()
        
        row = {
            "Month": m,
            "Sample Count": len(texts),
            "Ground Truth Pos %": f"{np.mean(ground_truth)*100:.2f}%"
        }
        
        for name, model in models.items():
            preds = model.predict(texts)
            pos_ratio = np.mean(preds)
            row[f"{name} Pos %"] = f"{pos_ratio*100:.2f}%"
            
        results.append(row)
        
    res_df = pd.DataFrame(results)
    return res_df


if __name__ == "__main__":
    timeline_path = "/Users/macm2/.gemini/antigravity/scratch/ml-vaccine-sentiment/data/processed/vaccine_doses_timeline.csv"
    fig1_path = "/Users/macm2/.gemini/antigravity/scratch/ml-vaccine-sentiment/report/figures/figure1_sentiment_trend.png"
    plot_sentiment_vs_vaccine_doses(timeline_path, fig1_path)
