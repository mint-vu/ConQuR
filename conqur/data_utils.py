import random

import datasets
import transformers


def get_tokenizer(model):
    return transformers.AutoTokenizer.from_pretrained(model, use_fast=False)


def get_wikitext2(nsamples, seed, seqlen, model, eval_mode=False):
    tokenizer = get_tokenizer(model)

    if eval_mode:
        testdata = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        return tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    traindata = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader


def get_c4(seqlen, model):
    tokenizer = get_tokenizer(model)
    valdata = datasets.load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )
    valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    return TokenizerWrapper(valenc)


def get_ptb(model):
    tokenizer = get_tokenizer(model)
    testdata = datasets.load_dataset('ptb_text_only', 'penn_treebank', split='test')
    return tokenizer(" ".join(testdata['sentence']), return_tensors='pt')


def get_calib_loader(name, nsamples=128, seed=0, seqlen=2048, model=''):
    """Calibration data for GPTQ."""
    if name != 'wikitext2':
        raise ValueError(f'Only wikitext2 calibration is supported, got {name}')
    return get_wikitext2(nsamples, seed, seqlen, model, eval_mode=False)


def get_eval_loader(name, seed=0, seqlen=2048, model=''):
    """Test split used for perplexity evaluation."""
    if name == 'wikitext2':
        return get_wikitext2(None, seed, seqlen, model, eval_mode=True)
    if name == 'ptb':
        return get_ptb(model)
    if name == 'c4':
        return get_c4(seqlen, model)
    raise ValueError(f'Unknown dataset {name}')
