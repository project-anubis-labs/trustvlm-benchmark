import torch

def get_token_ids(processor, word):
    # Add leading space → matches generation behavior
    word = " " + word.strip()
    
    tokens = processor.tokenizer(
        word,
        add_special_tokens=False
    ).input_ids
    
    return tokens

import torch

def compute_sequence_probability(model, processor, device, image, prompt, token_ids):
    """
    Computes P(token1, token2, ...) correctly using autoregressive conditioning
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

        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)

        prob = probs[0, token_id]
        total_log_prob += torch.log(prob + 1e-12)

        # Append token to input for next step
        next_token = torch.tensor([[token_id]]).to(device)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token], dim=1)

    return torch.exp(total_log_prob).item()