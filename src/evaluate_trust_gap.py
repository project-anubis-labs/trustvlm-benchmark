import torch
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from tqdm import tqdm
import os

# --------------------------------------------------
# ⚙️ CONFIG
# --------------------------------------------------
DEVICE = "cuda"
MODEL_ID = "llava-hf/llava-1.5-13b-hf"

DATA_PATH = "data/cfcs_pilot_dataset/dataset_metadata.csv"
OUTPUT_PATH = "results/trust_gap_results.csv"

# --------------------------------------------------
# 🧠 LOAD MODEL
# --------------------------------------------------
print("Loading model...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).to(DEVICE)

model.eval()

# --------------------------------------------------
# 🔑 TOKENIZATION (FIXED)
# --------------------------------------------------
def get_token_ids(word):
    # IMPORTANT: leading space for LLaMA tokenizer
    word = " " + word.strip()
    
    tokens = processor.tokenizer(
        word,
        add_special_tokens=False
    ).input_ids
    
    return tokens


# --------------------------------------------------
# 🔥 CORE: SEQUENCE LOG PROBABILITY
# --------------------------------------------------
def compute_log_prob(image, prompt, token_ids):
    """
    Computes:
    log P(token1, token2, ...)
    using autoregressive conditioning
    """

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(DEVICE)

    total_log_prob = 0.0

    for token_id in token_ids:
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)

        prob = probs[0, token_id]

        # log probability (stable)
        log_prob = torch.log(prob + 1e-12)
        total_log_prob += log_prob.item()

        # append token → autoregressive step
        next_token = torch.tensor([[token_id]]).to(DEVICE)
        inputs["input_ids"] = torch.cat(
            [inputs["input_ids"], next_token],
            dim=1
        )

    return total_log_prob


# --------------------------------------------------
# 🧠 TRUST GAP FUNCTION
# --------------------------------------------------
def compute_trust_gap(image_path, query, visual_word, prior_word):
    image = Image.open(image_path).convert("RGB")

    # improved prompt (important)
    prompt = f"<image>\n{query} Answer in one word."

    visual_ids = get_token_ids(visual_word)
    prior_ids = get_token_ids(prior_word)

    log_prob_visual = compute_log_prob(image, prompt, visual_ids)
    log_prob_prior = compute_log_prob(image, prompt, prior_ids)

    trust_gap = log_prob_prior - log_prob_visual

    hallucinated = 1 if trust_gap > 0 else 0

    return {
        "log_prob_visual": log_prob_visual,
        "log_prob_prior": log_prob_prior,
        "trust_gap": trust_gap,
        "hallucinated": hallucinated
    }


# --------------------------------------------------
# 🚀 MAIN LOOP
# --------------------------------------------------
def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    results = []

    print("Running evaluation...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            res = compute_trust_gap(
                row["filepath"],
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

    results_df = pd.DataFrame(results)

    os.makedirs("results", exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n✅ Saved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()