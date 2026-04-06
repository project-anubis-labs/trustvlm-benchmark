from PIL import Image
from token_utils import get_token_ids, compute_sequence_log_prob

PROMPT_TEMPLATE = "<image>\n{query} Answer in one word."


def compute_trust_gap(model, processor, device, image_path, query, visual_word, prior_word):
    """
    Returns log P(visual_word | image, query) and log P(prior_word | image, query),
    the trust gap (log_prob_prior - log_prob_visual), and a hallucination flag.
    """
    image = Image.open(image_path).convert("RGB")
    prompt = PROMPT_TEMPLATE.format(query=query)

    visual_ids = get_token_ids(processor, visual_word)
    prior_ids = get_token_ids(processor, prior_word)

    log_prob_visual = compute_sequence_log_prob(model, processor, device, image, prompt, visual_ids)
    log_prob_prior = compute_sequence_log_prob(model, processor, device, image, prompt, prior_ids)

    trust_gap = log_prob_prior - log_prob_visual

    return {
        "log_prob_visual": log_prob_visual,
        "log_prob_prior": log_prob_prior,
        "trust_gap": trust_gap,
        "hallucinated": 1 if trust_gap > 0 else 0
    }
