"""Load model / training base parameters from a YAML config (default: configs/LEAF_mpnet.yaml).

The YAML holds the *static* base parameters shared across stages plus the per-stage
training hyper-parameters. Per-run fields (model dims, embedding model) are still
supplied on the command line via --dim/--emb and merged in here, so the
checkpoint-path fingerprint stays driven by the CLI rather than the YAML.

This module also loads the auxiliary configs: per-task definitions
(configs/tasks.yaml) holding the dataset labels and instruction phrasings, and the
environment config (configs/env.yaml) holding the embedding-model registry and
data-folder locations.
"""
from pathlib import Path

import yaml
from transformers import PretrainedConfig

# Anchor config paths to this file's location so they resolve regardless of CWD.
_CONFIG_DIR = Path(__file__).resolve().parent / 'configs'
DEFAULT_CONFIG = _CONFIG_DIR / 'LEAF_mpnet.yaml'
DEFAULT_TASKS = _CONFIG_DIR / 'tasks.yaml'
DEFAULT_ENV = _CONFIG_DIR / 'env.yaml'


def load_yaml(path=DEFAULT_CONFIG):
    with open(path) as f:
        return yaml.safe_load(f)


def load_targets(path=DEFAULT_TASKS):
    """Dataset -> list of class labels (the candidate answer space)."""
    return load_yaml(path)['datasets']


def load_instructions(path=DEFAULT_TASKS):
    """Task family -> list of instruction phrasings."""
    return load_yaml(path)['instructions']


def load_text_embeddings(path=DEFAULT_ENV):
    """Model name -> {dim, path, loader?} registry for text-embedding models.

    Each entry in env.yaml is a one-line "<name> <dim> <source> [loader]" string:
    `name` is the short id, `source` the HuggingFace repo id (loaded with
    SentenceTransformer), and the optional `loader` ('cls') selects the dedicated
    BERT [CLS] loader instead.
    """
    registry = {}
    for line in load_yaml(path)['embedding_models']:
        name, dim, source, *rest = line.split()
        spec = {'dim': int(dim), 'path': source}
        if rest:
            spec['loader'] = rest[0]
        registry[name] = spec
    return registry


def load_paths(path=DEFAULT_ENV):
    """Data-folder locations (raw_data / pretrain / downstream) as a dict of Path."""
    return {key: Path(value) for key, value in load_yaml(path)['paths'].items()}


# Output dimension of each supported text-embedding model (derived from the registry).
Embedding_Dimension = {name: spec['dim'] for name, spec in load_text_embeddings().items()}


def build_model_config(cfg, dim=None, emb=None):
    """Build the model PretrainedConfig from the YAML `model:` section.

    The `model:` block is the full default model (static base + per-run dims +
    embedding model). `dim` and `emb`, when given, override those defaults — used
    for CLI ablation sweeps:
      dim = 'win-cnn-token-lay'            (pretrain Tower) or
            'win-cnn-token-lay-numq-qlay'  (full LEAF model)
    text_emb_model_dim is always derived from the embedding registry.
    """
    m = dict(cfg['model'])
    if dim is not None:
        parts = dim.split('-')
        m.update(window_len=int(parts[0]), dim_cnn=int(parts[1]),
                 dim_token=int(parts[2]), num_layers=int(parts[3]))
        if len(parts) >= 6:
            m.update(num_q=int(parts[4]), num_qformer_layers=int(parts[5]))
    emb = emb or m.get('text_emb_model_name')
    if emb is not None:
        m['text_emb_model_name'] = emb
        m['text_emb_model_dim'] = Embedding_Dimension[emb]
    return PretrainedConfig(**m)


def build_train_config(cfg, stage):
    """Return the PretrainedConfig for one training stage ('pretrain'/'instruct'/'downstream')."""
    return PretrainedConfig(**cfg[stage])
