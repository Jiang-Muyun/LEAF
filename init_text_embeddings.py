"""
Generate and cache text embeddings for LEAF labels and instructions.

Copyright 2022 Centre for Brain Computing Research (CBCR), College of Computing and Data Science (CCDS), Nanyang Technological University (NTU);
licensed under the CBCR License 1.0 (see LICENSE).
"""

import argparse
from pathlib import Path

import numpy as np

from load_config import load_text_embeddings, load_targets, load_instructions

TEXT_EMBEDDINGS = load_text_embeddings()
EMBEDDING_ROOT = Path('text_embeddings')


def _paths(model_name):
    """The (txt, npy) cache paths for one model."""
    return EMBEDDING_ROOT / f'{model_name}.txt', EMBEDDING_ROOT / f'{model_name}.npy'


def load_embeddings(model_name):
    """Read the cached (txt, npy) pair -> {text: np.ndarray(dim,) float32}.

    The .txt holds one text per line and the .npy holds a (N, dim) array whose
    row i is the embedding of line i, so the two are aligned by position.
    """
    txt_path, npy_path = _paths(model_name)
    texts = txt_path.read_text().splitlines()
    embs = np.load(npy_path).astype(np.float32)
    assert len(texts) == embs.shape[0], \
        f'{model_name}: {len(texts)} texts != {embs.shape[0]} embeddings'
    return {text: embs[i] for i, text in enumerate(texts)}


class LEAF_Text_Embedding:
    def __init__(self, model_name):
        # Heavy ML deps imported lazily so load_embeddings() stays import-cheap.
        import torch
        from transformers import BertTokenizer, BertModel
        from sentence_transformers import SentenceTransformer
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        spec = TEXT_EMBEDDINGS[model_name]
        self.model_name = model_name
        self.dimension = spec['dim']

        if spec.get('loader') == 'cls':
            tokenizer = BertTokenizer.from_pretrained(spec['path'])
            model = BertModel.from_pretrained(spec['path']).to(device)
            model.requires_grad_(False)
            model.eval()
            def _bert_call(text):
                inputs = {k: v.to(device) for k, v in tokenizer(text, return_tensors="pt").items()}
                with torch.no_grad():
                    outputs = model(**inputs)
                emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                return (emb / np.linalg.norm(emb)).astype('float32')
            self.call = _bert_call

        else:
            model = SentenceTransformer(spec['path'], device=device)
            self.call = lambda text: model.encode(text, normalize_embeddings=True).astype('float32')

    def get_embedding(self, text):
        return self.call(text)


def all_texts():
    """All texts to embed (target labels then instruction phrasings), deduplicated
    while preserving the order they appear in the config files."""
    texts = []
    for sentences in load_targets().values():
        texts.extend(sentences)
    for sentences in load_instructions().values():
        texts.extend(sentences)
    return list(dict.fromkeys(texts))  # dedupe, keep first-occurrence order


def cache_embeddings(model_name):
    """Generate / extend the (txt, npy) cache for one model.

    Texts are kept in their config order. On a re-run, any texts already cached
    keep their position and only NEW texts are embedded and appended to the end
    (so adding an instruction doesn't recompute or reorder existing rows).

    Writes text_embeddings/<model>.txt (one text per line) and
    text_embeddings/<model>.npy (float32, shape (N, dim); row i = line i).
    Returns (txt_path, npy_path).
    """
    texts = all_texts()
    txt_path, npy_path = _paths(model_name)

    if txt_path.exists() and npy_path.exists():
        existing = txt_path.read_text().splitlines()
        existing_embs = np.load(npy_path).astype(np.float32)
        assert len(existing) == existing_embs.shape[0], \
            f'{model_name}: {len(existing)} texts != {existing_embs.shape[0]} embeddings'
    else:
        existing, existing_embs = [], None

    seen = set(existing)
    new_texts = [t for t in texts if t not in seen]
    if not new_texts:
        print(f"{model_name}: up to date ({len(existing)} texts cached, nothing new).")
        return txt_path, npy_path

    embed = LEAF_Text_Embedding(model_name)
    print(f"Model: {model_name}, dim: {embed.dimension}")
    print(f"{len(existing)} cached; embedding {len(new_texts)} new text(s), appending to the end.")

    new_embs = np.zeros((len(new_texts), embed.dimension), dtype=np.float32)
    for i, text in enumerate(new_texts):
        print(f"Processing: {text}")
        new_embs[i] = embed.get_embedding(text)

    # Defensive L2 re-normalization in float32: some models normalize internally in
    # reduced precision (e.g. bf16), leaving stored norms off by ~1e-3.
    new_embs /= np.linalg.norm(new_embs, axis=1, keepdims=True)

    out_texts = existing + new_texts
    out_embs = new_embs if existing_embs is None else np.concatenate([existing_embs, new_embs], axis=0)

    EMBEDDING_ROOT.mkdir(parents=True, exist_ok=True)
    txt_path.write_text('\n'.join(out_texts) + '\n')
    np.save(npy_path, out_embs)
    print(f"Saved {txt_path} ({len(out_texts)} lines) and {npy_path} {out_embs.shape}")
    return txt_path, npy_path


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('model', type=str, choices=list(TEXT_EMBEDDINGS.keys()))
    args = p.parse_args()
    cache_embeddings(args.model)
