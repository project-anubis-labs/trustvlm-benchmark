import torch


def get_token_ids(processor, word):
    # Leading space matches LLaMA tokenizer's generation behaviour
    word = " " + word.strip()
    tokens = processor.tokenizer(word, add_special_tokens=False).input_ids
    return tokens


def compute_sequence_log_prob(model, processor, device, image, prompt, token_ids):
    """
    Computes log P(token_1, token_2, ...) via autoregressive conditioning.
    Returns a scalar float (sum of per-token log-probs).
    """
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(device)

    total_log_prob = 0.0

    for token_id in token_ids:
        with torch.no_grad():
            outputs = model(**inputs)

        # log_softmax is numerically stable; avoids softmax + log + epsilon
        log_probs = torch.log_softmax(outputs.logits[:, -1, :], dim=-1)
        total_log_prob += log_probs[0, token_id].item()

        # Append the token and extend the attention mask to match
        next_token = torch.tensor([[token_id]], dtype=torch.long, device=device)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token], dim=1)
        inputs["attention_mask"] = torch.cat(
            [inputs["attention_mask"], torch.ones((1, 1), dtype=torch.long, device=device)],
            dim=1
        )

    return total_log_prob
