import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from tqdm import tqdm

from model_loader import load_model
from evaluate_trust_gap import compute_trust_gap

# --------------------------------------------------
# Paths
# --------------------------------------------------
DATA_PATH = "data/cfcs_pilot_dataset/dataset_metadata.csv"
# The CSV stores filepaths relative to the data/ dir (e.g. "cfcs_pilot_dataset/image_001.png")
DATA_DIR = "data"
OUTPUT_PATH = "results/trust_gap_results.csv"
AGG_OUTPUT_PATH = "results/aggregated_results.csv"


# --------------------------------------------------
# Evaluation loop
# --------------------------------------------------
def run_evaluation(model, processor, device, df):
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # Prepend DATA_DIR so the path resolves from the project root
            image_path = os.path.join(DATA_DIR, row["filepath"])
            res = compute_trust_gap(
                model,
                processor,
                device,
                image_path,
                row["vqa_query"],
                row["target_visual"],
                row["target_prior"]
            )
            results.append({
                "image_id": row["image_id"],
                "category": row["category"],
                "visual_word": row["target_visual"],
                "prior_word": row["target_prior"],
                "log_prob_visual": res["log_prob_visual"],
                "log_prob_prior": res["log_prob_prior"],
                "trust_gap": res["trust_gap"],
                "hallucinated": res["hallucinated"]
            })
        except Exception as e:
            print(f"Error processing {row['image_id']}: {e}")
    return results


# --------------------------------------------------
# Aggregation
# --------------------------------------------------
def aggregate_results(df):
    agg = df.groupby("category").agg(
        avg_trust_gap=("trust_gap", "mean"),
        std_trust_gap=("trust_gap", "std"),
        avg_log_prob_visual=("log_prob_visual", "mean"),
        avg_log_prob_prior=("log_prob_prior", "mean"),
    ).reset_index()

    win_rate = df.groupby("category").apply(
        lambda x: (x["trust_gap"] > 0).mean()
    ).reset_index(name="prior_win_rate")

    return agg.merge(win_rate, on="category")


# --------------------------------------------------
# Entry point
# --------------------------------------------------
def main():
    print("Loading model...")
    model, processor, device = load_model()

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print("Running evaluation...")
    results = run_evaluation(model, processor, device, df)

    results_df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved per-image results to {OUTPUT_PATH}")

    print("\nAggregated results:")
    agg = aggregate_results(results_df)
    print(agg.to_string(index=False))
    agg.to_csv(AGG_OUTPUT_PATH, index=False)
    print(f"Saved aggregated results to {AGG_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
