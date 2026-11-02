import math

import torch
import torch.nn as nn
from tqdm import tqdm


def _get_first_param_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device('cpu')


@torch.no_grad()
def ppl_evaluator(model, testenc, dev, batch_size=1):
    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False

    # Convert the whole text of evaluation dataset into batches of sequences.
    input_ids = testenc.input_ids  # (1, text_len)
    nsamples = input_ids.numel() // model.seqlen
    input_ids = input_ids[:, :nsamples * model.seqlen].view(nsamples, model.seqlen)  # (nsamples, seqlen)
    input_ids = [input_ids[i:i + batch_size] for i in range(0, nsamples, batch_size)]

    nlls = []
    input_device = dev if dev is not None else _get_first_param_device(model)
    loss_fct = nn.CrossEntropyLoss()

    pbar = tqdm(range(len(input_ids)), desc="Evaluating PPL")
    for i in pbar:
        batch = input_ids[i].to(input_device)
        hidden_states = model.model(batch)[0]
        logits = model.lm_head(hidden_states)
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[i][:, 1:].to(shift_logits.device)
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        neg_log_likelihood = loss.float()
        if not math.isnan(neg_log_likelihood):
            nlls.append(neg_log_likelihood)
        if nlls:
            pbar.set_description(f"PPL: {torch.exp(torch.stack(nlls).mean()).item():.2f}")

    model.config.use_cache = use_cache
    return torch.exp(torch.stack(nlls).mean()).item()
