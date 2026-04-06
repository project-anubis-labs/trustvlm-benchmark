import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from tqdm import tqdm

from model_loader import load_model
from evaluate_trust_gap import compute_trust_gap

# Paths
DATA_PATH = "data/cfcs_pilot_dataset/dataset_metadata.csv"
OUTPUT_PATH = "results/trust_gap_results.csv"

def main():
    print("Loading model...")
    model, processor, device = load_model()

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    results = []

    print("Running evaluation...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            res = compute_trust_gap(
                model,
                processor,
                device,
                row["filepath"],
                row["vqa_query"],
                row["target_visual"],
                row["target_prior"]
            )

            results.append({
    "image_id": row["image_id"],
    "category": row["category"],
    "p_visual": res["p_visual"],
    "p_prior": res["p_prior"],
    "trust_gap": res["trust_gap"]
})

        except Exception as e:
            print(f"Error processing {row['image_id']}: {e}")

    results_df = pd.DataFrame(results)

    os.makedirs("results", exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved results to {OUTPUT_PATH}")

import pandas as pd

df = pd.read_csv("results/trust_gap_results.csv")

# Per category (physics, scale, etc.)
agg = df.groupby("category").agg({
    "trust_gap": ["mean", "std"],
    "p_prior": "mean",
    "p_visual": "mean"
}).reset_index()

agg.columns = ["category", "avg_trust_gap", "std_trust_gap", "avg_p_prior", "avg_p_visual"]

# Win rate (how often prior wins)
win_rate = df.groupby("category").apply(
    lambda x: (x["trust_gap"] > 0).mean()
).reset_index(name="prior_win_rate")

agg = agg.merge(win_rate, on="category")

print(agg)
agg.to_csv("results/aggregated_results.csv", index=False)


if __name__ == "__main__":
    main()