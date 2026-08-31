import torch


def embed_text(tokenizer, model, text):
    """Tokenize text and look up its embeddings from the model's own
    embedding table. Manual equivalent of what generate(input_ids=...)
    does internally — needed explicitly here because we must
    concatenate with BEV tokens *before* the model ever sees the
    sequence, and generate() has no way to accept a mix of discrete
    token ids and pre-computed vectors in one call."""
    ids = tokenizer(text, return_tensors="pt")["input_ids"].to(model.device)
    embeds = model.get_input_embeddings()(ids)
    return ids, embeds


def fuse_bev_and_text_embeds(bev_tokens, text_embeds):
    """Concatenate BEV tokens (as a prefix) with text embeddings, and
    build the matching attention mask. This is the one fusion step
    used everywhere BEV tokens get combined with a prompt — Stage 4's
    single generation, Stage 5's real/zeroed/shuffled comparison, and
    training's forward pass all go through this same function."""
    combined_embeds = torch.cat([bev_tokens, text_embeds], dim=1)
    combined_mask = torch.ones(
        combined_embeds.shape[:2], dtype=torch.long, device=combined_embeds.device
    )
    return combined_embeds, combined_mask
