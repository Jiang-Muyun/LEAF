"""
Define the core neural network components of the LEAF model.

Copyright 2022 Centre for Brain Computing Research (CBCR), College of Computing and Data Science (CCDS), Nanyang Technological University (NTU);
licensed under the CBCR License 1.0 (see LICENSE).
"""

import torch
import einops
import random
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from transformers import PretrainedConfig

class DebugPrint(nn.Module):
    def __init__(self, text):
        super().__init__()
        self.text = text

    def forward(self, x):
        print(self.text, x.shape)
        return x

class EmbeddingClassifier(nn.Module):
    def __init__(self, dim_emb, n_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(dim_emb, dim_emb//2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(dim_emb//2, dim_emb//4),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(dim_emb//4, n_classes),
        )
    def forward(self, x):
        return self.classifier(x)

class FlattenClassifier(nn.Module):
    def __init__(self, dim_token, n_tokens, n_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            Rearrange('B N D -> B (N D)'),
            nn.Linear(n_tokens * dim_token, 2 * dim_token),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(2 * dim_token, dim_token),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(dim_token, n_classes),
        )
    def forward(self, x):
        return self.classifier(x)

class Tokenizer(nn.Module):
    # Splits raw EEG into fixed-length windows and encodes each window into a token
    # via a temporal-spatial-style CNN
    def __init__(self, n_channels=65, dim_cnn=64, dim_token=128, window_len=100, pool1=10, dropout=0.5):
        super().__init__()
        self.n_channels = n_channels
        self.dim_cnn = dim_cnn
        self.dim_token = dim_token
        self.window_len = window_len
        self.pool1 = pool1
        assert window_len % pool1 == 0, f"window_len {window_len} must be divisible by {pool1}"

        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, dim_cnn, (1, 40), padding=(0, 20)),       # temporal filter per channel
            nn.Conv2d(dim_cnn, dim_cnn, (n_channels, 1)),          # spatial filter across channels
            nn.SyncBatchNorm(dim_cnn),
            nn.GELU(),
            nn.MaxPool2d((1, pool1), stride=(1, pool1)),           # downsample time axis
            nn.Dropout(dropout),
        )
        self.out_dim = (dim_cnn *  (window_len // pool1))          # flattened token dim

    def forward(self, x):
        # x: (B, C, T) → unfold into windows → encode each window → (B, N, out_dim)
        x = x.unfold(-1, self.window_len, self.window_len)          # (B, C, N, window_len)
        B, C, N, T = x.shape
        x = einops.rearrange(x, "B C N T -> (B N) 1 C T")          # batch windows
        x = self.enc_conv(x)
        x = einops.rearrange(x, '(B N) F 1 T -> B N (F T)', B=B, N=N)  # flatten to tokens
        return x
    
class TransformerLayer(nn.Module):
    # Standard pre-norm transformer layer: norm → self-attn → residual, norm → FFN → residual.
    def __init__(self, embed_dim, num_heads, dim_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(embed_dim, dim_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ff, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=True)
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=True)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.gelu

    # self-attention block
    def _sa_block(self,x,attn_mask,key_padding_mask,is_causal = False):
        x = self.self_attn(x, x, x, 
                           attn_mask=attn_mask, 
                           key_padding_mask=key_padding_mask,
                           need_weights=False,
                           is_causal=is_causal)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        x = src
        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
        x = x + self._ff_block(self.norm2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, dim_token, num_layers, ff_scale=4, num_heads=8, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(dim_token, num_heads, dim_ff=dim_token * ff_scale, dropout=dropout,
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim_token, elementwise_affine=True)

    def forward(self, tokens, mask=None):
        x = tokens
        for layer in self.layers:
            x = layer(x, src_mask=mask)
        return self.norm(x)

class Tower(nn.Module):
    # Self-supervised pre-training backbone with two branches:
    #   mask_branch : masked token prediction (bidirectional, like BERT)
    #   next_branch : next-window prediction  (causal, like GPT)
    # Both branches share the tokenizer and pos_embedding but have independent
    # projection layers and transformers. At inference, forward() concatenates
    # both branches' outputs as the token representation.
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.max_tokens = (config.max_seq_len - config.window_len) // config.window_len + 1

        self.tokenizer = Tokenizer(config.n_channels, config.dim_cnn, config.dim_token, config.window_len)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.max_tokens, config.dim_token))

        self.freq_cutoff_min = config.freq_cutoff_min
        self.freq_cutoff_max = config.freq_cutoff_max
        self.freq_cutoff_bandwidth = config.freq_cutoff_bandwidth

        # ---- mask branch (bidirectional) ----
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.dim_token))
        self.mask_inProj = nn.Linear(self.tokenizer.out_dim, config.dim_token)
        self.mask_transformer = Transformer(config.dim_token, config.num_layers, config.ff_scale, config.num_heads, config.dropout)
        self.mask_outProj = nn.Sequential(
            nn.Linear(config.dim_token, config.n_channels * 20),
            nn.GELU(),
            Rearrange('B N (C T) -> B N C T', C=config.n_channels),
            nn.Linear(20, config.window_len),
            Rearrange('B N C T -> B C (N T)'),
        )

        # ---- next branch (causal) ----
        self.register_buffer('causal_mask', torch.triu(torch.full((self.max_tokens, self.max_tokens), float("-inf")), diagonal=1))
        self.next_inProj = nn.Linear(self.tokenizer.out_dim, config.dim_token)
        self.next_transformer = Transformer(config.dim_token, config.num_layers, config.ff_scale, config.num_heads, config.dropout)
        self.next_outProj = nn.Sequential(
            nn.Linear(config.dim_token, config.n_channels * 20),
            nn.GELU(),
            Rearrange('B N (C T) -> B N C T', C=config.n_channels),
            nn.Linear(20, config.window_len),
            Rearrange('B N C T -> B C (N T)'),
        )

    @torch.no_grad()
    def frequency_cutoff(self, x):
        # Data augmentation: zero out a random frequency band [low, low+bandwidth] Hz.
        # Forces the model to learn representations not reliant on any single frequency band.
        low = random.randint(self.freq_cutoff_min, self.freq_cutoff_max)
        high = low + self.freq_cutoff_bandwidth
        with torch.amp.autocast('cuda', enabled=False):
            x32 = x.float()
            Xf = torch.fft.rfft(x32, dim=-1)
            freqs = torch.fft.rfftfreq(x32.size(-1), d=1./self.config.sfreq)
            m = (freqs >= low) & (freqs < high)
            Xf[..., m] = 0
            x_corrupt = torch.fft.irfft(Xf, n=x.size(-1), dim=-1).type_as(x)
        return x_corrupt

    def mask_branch(self, token_aug, x):
        # BERT-style masked token prediction: randomly replace tokens with a learned
        # mask token, reconstruct the original signal window with MSE.
        B, N, D = token_aug.shape
        mask = torch.rand((B, N), device=token_aug.device) < self.config.mask_ratio
        inp = self.mask_inProj(token_aug.clone())
        inp[mask] = self.mask_token.to(inp.dtype)
        pred = self.mask_transformer(inp + self.pos_embedding[:, :N])
        x_rec = self.mask_outProj(pred)
        return F.mse_loss(x_rec, x)

    def next_branch(self, token_aug, x):
        # Strict next-window prediction: state i can see corrupted tokens 0..i
        # and is decoded against the clean signal window i+1. Keeping the same
        # token positions here and in forward() also avoids a train/inference
        # positional-embedding shift.
        B, N, D = token_aug.shape
        if N < 2:
            raise ValueError(f'next-window prediction requires at least 2 windows, got {N}')
        inp = self.next_inProj(token_aug.clone())
        pred = self.next_transformer(
            inp + self.pos_embedding[:, :N],
            mask=self.causal_mask[:N, :N]
        )
        x_rec = self.next_outProj(pred[:, :-1])
        x_next = x[..., self.config.window_len:N * self.config.window_len]
        return F.mse_loss(x_rec, x_next)

    def loss(self, x):
        # Apply frequency-cutoff augmentation once, then compute both branch losses.
        x_aug = self.frequency_cutoff(x)
        token_aug = self.tokenizer(x_aug)
        loss_S = self.mask_branch(token_aug, x)
        loss_T = self.next_branch(token_aug, x)
        return loss_S, loss_T

    def forward(self, x):
        # At inference: encode with both branches and concatenate along token dim.
        # Output: (B, 2N, dim_token) — doubled sequence used as Q-Former input.
        tokens = self.tokenizer(x)
        B, N, D = tokens.shape
        outS = self.mask_transformer(
            self.mask_inProj(tokens) + self.pos_embedding[:, :N],
        )
        outT = self.next_transformer(
            self.next_inProj(tokens) + self.pos_embedding[:, :N],
            mask=self.causal_mask[:N, :N]
        )
        return torch.cat([outS, outT], dim=1)

class QFormerLayer(nn.Module):
    def __init__(self, dim_q, dim_kv, n_heads, ff_scale=4):
        super().__init__()
        # self-attn (queries attend to queries)
        self.self_attn = nn.MultiheadAttention(dim_q, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim_q)

        self.cross_attn = nn.MultiheadAttention(
            dim_q, n_heads, batch_first=True,
            kdim=dim_kv, vdim=dim_kv
        )
        self.norm2 = nn.LayerNorm(dim_q)

        self.ff = nn.Sequential(
            nn.Linear(dim_q, ff_scale * dim_q),
            nn.GELU(),
            nn.Linear(ff_scale * dim_q, dim_q)
        )
        self.norm3 = nn.LayerNorm(dim_q)

    def forward(self, q, kv, return_weights=False):
        """
        q:  (B, num_q, dim_q)
        kv: (B, T, dim_kv)
        """
        # ---- Self-attention on queries (pre-norm) ----
        q_self, attn_self = self.self_attn(
            self.norm1(q), self.norm1(q), self.norm1(q),
            need_weights=True,
            average_attn_weights=True
        )
        q = q + q_self

        # ---- Cross-attention: queries attend to input tokens (pre-norm) ----
        q_cross, attn_cross = self.cross_attn(
            self.norm2(q), kv, kv,
            need_weights=True,
            average_attn_weights=True
        )
        q = q + q_cross

        # ---- FFN (pre-norm) ----
        q = q + self.ff(self.norm3(q))

        if return_weights:
            return q, {
                "self_attn": attn_self,     # (B, num_q, num_q)
                "cross_attn": attn_cross,   # (B, num_q, T)
            }

        return q

class TextConditionedQFormer(nn.Module):
    def __init__(self, 
                 num_q=8,          # number of queries
                 dim_q=256,        # query dimension
                 dim_kv=256,       # token dimension
                 n_heads=8,
                 n_layers=4,
                 ff_scale=4,
                 text_dim=768,  # instruction embedding dimension
                 max_tokens=40, # number of input tokens
                 ):
        super().__init__()
        self.q_len = num_q
        self.dim_q = dim_q

        # learnable queries
        self.base_queries = nn.Parameter(torch.randn(num_q, dim_q) / dim_q**0.5)

        self.max_pos = max_tokens * 2
        self.pos_embedding = nn.Parameter(torch.randn(1, self.max_pos, dim_kv))  # fix: dim_kv not dim_q

        # FiLM modulation (instruct -> gamma, beta)
        # fix: output dim must match dim_kv (tokens), not dim_q (queries)
        self.film = nn.Linear(text_dim, 2 * dim_kv)

        # Q-Former layer stack
        self.layers = nn.ModuleList([
            QFormerLayer(dim_q, dim_kv, n_heads, ff_scale) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(dim_q)  # needed: pre-norm leaves last layer output unnormalized

        self.output_embed = nn.Sequential(
            Rearrange('B N D -> B (N D)'),
            nn.Dropout(0.3),
            nn.Linear(dim_q * num_q, text_dim),
        )


    def forward(self, tokens, instruct, return_all_q=False):
        B, N, D = tokens.shape
        assert N <= self.max_pos, f"Token sequence length {N} exceeds pos_embedding size {self.max_pos}"

        # FiLM: condition token features on the instruction embedding (unconstrained).
        gb = self.film(instruct)                          # (B, 2*dim_kv)
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1)                        # (B,1,dim_kv)
        beta  = beta.unsqueeze(1)                         # (B,1,dim_kv)
        tokens = (1 + gamma) * tokens + beta              # scale+shift each token (B,N,dim_kv)
        tokens = tokens + self.pos_embedding[:, :N]       # add positional encoding

        # Expand shared learnable queries to the batch dimension.
        q = self.base_queries.unsqueeze(0).expand(B, -1, -1)  # (B, num_q, dim_q)

        all_q = [q]  # layer 0 = initial queries before any attention
        for layer in self.layers:
            q = layer(q, tokens)
            all_q.append(q)

        q = self.final_norm(q)                            # normalize pre-norm leaves last layer unnormalized
        output = self.output_embed(q)                     # flatten + project → (B, text_dim)

        if return_all_q:
            # all_q: list of (B, num_q, dim_q), len = n_layers+1
            return output, all_q
        return output

class LEAF(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tower = Tower(config)
        self.qformer = TextConditionedQFormer(num_q=config.num_q, 
                                              dim_q=config.dim_token, 
                                              dim_kv=config.dim_token, 
                                              n_layers=config.num_qformer_layers,
                                              text_dim=config.text_emb_model_dim,
                                              max_tokens=self.tower.max_tokens)
        
    def forward(self, x, instruct_emb):
        tokens = self.tower(x)
        emb = F.normalize(self.qformer(tokens, instruct_emb), dim=-1)
        return emb, tokens
    
    def forward_with_q(self, x, instruct_emb):
        tokens = self.tower(x)
        out, all_q = self.qformer(tokens, instruct_emb, return_all_q=True)
        emb = F.normalize(out, dim=-1)
        return emb, tokens, all_q


if __name__ == "__main__":

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    config = PretrainedConfig(
        n_channels = 65,
        dim_cnn = 64,
        ff_scale = 4,
        num_heads = 8,
        num_layers = 12,
        dim_token = 256,
        max_seq_len=200 * 10,
        sfreq=200,
        window_len=100,
        dropout=0.1,
        mask_ratio=0.5,
        freq_cutoff_min = 1,
        freq_cutoff_max = 50,
        freq_cutoff_bandwidth=6,

        num_q = 8,
        num_qformer_layers = 4,
        text_emb_model_dim = 768,
        text_emb_model_name = 'bert-base-uncased',
    )

    B, n_classes = 2, 5
    x = torch.randn(B, config.n_channels, config.max_seq_len)
    instruct_emb = torch.randn(B, config.text_emb_model_dim)
    prototypes = F.normalize(torch.randn(n_classes, config.text_emb_model_dim), dim=-1).t()

    print("=== Tower (No QFormer) ===")
    tower = Tower(config)
    print(f"  Total:           {count_parameters(tower)/1e6:.3f} M")
    print(f"  tokenizer:       {count_parameters(tower.tokenizer)/1e6:.3f} M")
    print(f"  mask_inProj:     {count_parameters(tower.mask_inProj)/1e6:.3f} M")
    print(f"  mask_transformer:{count_parameters(tower.mask_transformer)/1e6:.3f} M")
    print(f"  mask_outProj:    {count_parameters(tower.mask_outProj)/1e6:.3f} M")
    tokens = tower(x)
    print(f"  tokens: {x.shape} -> {tokens.shape}")
    loss_S, loss_T = tower.loss(x)
    print(f"  loss_S={loss_S.item():.4f}, loss_T={loss_T.item():.4f}")

    print("\n=== LEAF (Full Model) ===")
    model = LEAF(config)
    print(f"  Total:   {count_parameters(model)/1e6:.3f} M")
    print(f"  tower:   {count_parameters(model.tower)/1e6:.3f} M")
    print(f"  qformer: {count_parameters(model.qformer)/1e6:.3f} M")
    emb, tokens = model(x, instruct_emb)
    print(f"  forward: emb={emb.shape}, tokens={tokens.shape}")
    logits = emb @ prototypes
    print(f"  logits:  {logits.shape}")
