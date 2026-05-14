"""
ggeur_prompt.py - open_clip based PromptFL implementation

Uses open_clip (ViT-B-16 etc.) directly, no HuggingFace CLIPModel needed.
The only requirement is the same .bin checkpoint used by GGEUR feature extraction.
"""

import re
import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

logger = logging.getLogger(__name__)


def to_display_name(name: str) -> str:
    name = name.replace("_", " ")
    name = re.sub(r"\s+", " ", name).strip()
    return name


class PromptLearner(nn.Module):
    """
    Learnable soft prompt for open_clip text encoder.

    Prompt structure: [BOS] [ctx x n_ctx] [class tokens] [EOS] [PAD...]
    Only self.ctx is trainable; token embeddings for class names are frozen buffers.
    """

    token_prefix: torch.Tensor   # (K, 1, d)       - BOS embedding
    token_suffix: torch.Tensor   # (K, *, d)        - class tokens + EOS + PAD
    tokenized_prompts: torch.Tensor  # (K, context_length) - full token ids for mask

    def __init__(
        self,
        clip_model,           # open_clip model instance
        tokenizer,            # open_clip tokenizer
        classnames: List[str],
        n_ctx: int = 16,
        template: str = "a photo of a {}",
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.n_cls = len(classnames)
        self.n_ctx = n_ctx
        self.device = device
        self.context_length = clip_model.context_length  # typically 77

        # embedding dimension from token_embedding weight
        d = clip_model.token_embedding.embedding_dim

        # learnable context vectors
        self.ctx = nn.Parameter(torch.randn(n_ctx, d, device=device) * 0.02)

        # tokenize "a photo of a {classname}" for each class
        display_names = [to_display_name(c) for c in classnames]
        texts = [template.format(c) for c in display_names]
        tokenized = tokenizer(texts).to(device)  # (K, context_length)

        with torch.no_grad():
            token_emb = clip_model.token_embedding(tokenized)  # (K, L, d)

        # prefix = BOS token only (position 0)
        token_prefix = token_emb[:, :1, :]       # (K, 1, d)
        # suffix = everything after BOS (class tokens + EOS + padding)
        token_suffix = token_emb[:, 1:, :]       # (K, L-1, d)

        self.register_buffer("token_prefix", token_prefix)
        self.register_buffer("token_suffix", token_suffix)
        self.register_buffer("tokenized_prompts", tokenized)

        logger.info(f"PromptLearner: n_cls={self.n_cls}, n_ctx={n_ctx}, d={d}, "
                    f"context_length={self.context_length}")

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            prompts:  (K, context_length, d)  - full prompt embeddings
            eot_pos:  (K,)                    - position of EOT token per class
        """
        K = self.n_cls
        ctx = self.ctx.unsqueeze(0).expand(K, -1, -1)  # (K, n_ctx, d)

        # [BOS] + [ctx] + [class + EOS + PAD]
        # total length must equal context_length (77)
        # suffix already has length (context_length - 1), so after inserting n_ctx
        # context tokens we need to trim the suffix accordingly
        suffix = self.token_suffix[:, :self.context_length - 1 - self.n_ctx, :]
        prompts = torch.cat([self.token_prefix, ctx, suffix], dim=1)  # (K, L, d)

        # EOT position: last non-zero token in original tokenized_prompts, shifted
        # by n_ctx because ctx tokens are inserted at position 1, pushing everything right.
        # open_clip uses 0 as padding; EOT is the highest token id (49407) in each row.
        eot_pos = self.tokenized_prompts.argmax(dim=-1) + self.n_ctx  # (K,)

        return prompts, eot_pos


class TextEncoder(nn.Module):
    """
    Wraps open_clip's text transformer to encode prompt embeddings.
    Mirrors what open_clip.CLIP.encode_text() does, but accepts pre-built embeddings.
    """

    def __init__(self, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device

    def forward(
        self,
        prompts: torch.Tensor,    # (K, L, d)
        eot_pos: torch.Tensor,    # (K,)
        clip_model,               # open_clip model
    ) -> torch.Tensor:
        """Returns L2-unnormalized text features of shape (K, embed_dim)."""
        K, L, d = prompts.shape

        # positional embedding
        x = prompts + clip_model.positional_embedding[:L]  # (K, L, d)

        # open_clip's Transformer.forward() handles seq/batch permutation internally;
        # pass x as batch-first (K, L, d) directly.
        attn_mask = None
        if hasattr(clip_model, 'attn_mask') and clip_model.attn_mask is not None:
            attn_mask = clip_model.attn_mask[:L, :L].to(x.device)
        x = clip_model.transformer(x, attn_mask=attn_mask)  # (K, L, d)
        x = clip_model.ln_final(x)                # (K, L, d)

        # Clamp eot_pos to valid range to avoid out-of-bounds indexing
        eot_pos = eot_pos.clamp(max=L - 1)
        # take features at EOT token position
        x = x[torch.arange(K, device=self.device), eot_pos]  # (K, d)

        # project to joint embedding space
        x = x @ clip_model.text_projection        # (K, embed_dim)
        return x


class CustomCLIP(nn.Module):
    """
    open_clip based CustomCLIP for PromptFL.

    CLIP backbone is fully frozen. Only PromptLearner.ctx is trainable.
    forward() accepts pre-extracted image features (not raw images),
    matching GGEUR's workflow where features are cached.
    """

    def __init__(
        self,
        clip_model,
        tokenizer,
        classnames: List[str],
        n_ctx: int = 16,
        template: str = "a photo of a {}",
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()

        self.clip_model = clip_model.to(device)
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.clip_model.eval()

        self.classnames = classnames
        self.n_ctx = n_ctx
        self.template = template
        self.device = device

        self.prompt_learner = PromptLearner(
            clip_model=self.clip_model,
            tokenizer=tokenizer,
            classnames=classnames,
            n_ctx=n_ctx,
            template=template,
            device=device,
        )
        self.text_encoder = TextEncoder(device=device)

        logger.info(f"CustomCLIP (open_clip): {len(classnames)} classes, n_ctx={n_ctx}")

    def get_text_features(self) -> torch.Tensor:
        """Returns L2-normalized text features (K, embed_dim)."""
        prompts, eot_pos = self.prompt_learner()
        text_feats = self.text_encoder(prompts, eot_pos, self.clip_model)
        return F.normalize(text_feats, dim=-1)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_features: (B, embed_dim) - pre-extracted, NOT necessarily normalized
        Returns:
            logits: (B, K)
        """
        img_feats = F.normalize(image_features.to(self.device), dim=-1)
        text_feats = self.get_text_features()                    # (K, embed_dim)
        logit_scale = self.clip_model.logit_scale.exp()
        return logit_scale * img_feats @ text_feats.t()

    @torch.no_grad()
    def clone_prompt_only(self):
        new_obj = CustomCLIP(
            clip_model=self.clip_model,
            tokenizer=self.prompt_learner.processor if hasattr(self.prompt_learner, 'processor') else None,
            classnames=self.classnames,
            n_ctx=self.n_ctx,
            template=self.template,
            device=self.device,
        )
        new_obj.prompt_learner.ctx.data = self.prompt_learner.ctx.data.clone()
        return new_obj

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        new_obj = CustomCLIP(
            clip_model=self.clip_model,
            tokenizer=None,
            classnames=self.classnames,
            n_ctx=self.n_ctx,
            template=self.template,
            device=self.device,
        )
        new_obj.prompt_learner.load_state_dict(
            copy.deepcopy(self.prompt_learner.state_dict()), strict=True
        )
        memo[id(self)] = new_obj
        return new_obj
