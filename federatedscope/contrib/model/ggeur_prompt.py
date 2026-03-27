import torch
import torch.nn as nn
from typing import Tuple, List, cast
from transformers import CLIPProcessor, CLIPModel
from transformers.modeling_attn_mask_utils import (
    _create_4d_causal_attention_mask,
    _prepare_4d_attention_mask,
)

from transformers.modeling_outputs import BaseModelOutput


import logging
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import re


def to_display_name(name: str) -> str:
    """Convert class name to display format by replacing underscores with spaces."""
    name = name.replace("_", " ")
    name = re.sub(r"\s+", " ", name).strip()
    return name


class PromptLearner(nn.Module):
    """
    Learnable prompt module for CLIP text encoder.
    
    This module creates learnable context tokens that are inserted between
    the prefix (BOS token) and suffix (class name + EOS) of the text prompt.
    Only the context tokens (self.ctx) are trainable, while prefix and suffix
    embeddings are frozen.
    """
    # Declare buffer types to avoid type checker treating them as Optional[Tensor]
    token_prefix: torch.Tensor
    token_suffix: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor

    def __init__(
        self,
        clip_model: CLIPModel,
        processor: CLIPProcessor,
        classnames: List[str],
        n_ctx: int = 32,
        template: str = "a photo of a {}",
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        # self.clip_model = clip_model
        self.processor = processor
        self.classnames = classnames
        self.n_cls: int = len(classnames)
        self.n_ctx: int = n_ctx
        self.device = device
        
        # Get text embedding dimension from CLIP model config
        d: int = int(clip_model.text_model.config.hidden_size)
        
        # Initialize learnable context tokens with small random values
        self.ctx = nn.Parameter(torch.randn(n_ctx, d, device=device) * 0.02)

        logger.info(f"PromptLearner initialized: n_cls={self.n_cls}, n_ctx={n_ctx}, hidden_size={d}, device={device}")

        # Step 1: Construct full text for each class (used to get suffix token ids and EOS position)
        display_class_names = [to_display_name(name) for name in classnames]
        texts = [template.format(name) for name in display_class_names]
        tok = processor(text=texts, padding=True, truncation=True, return_tensors="pt")  # type: ignore

        input_ids_local = cast(torch.Tensor, tok["input_ids"])  # (K, L)
        attention_mask_local = cast(torch.Tensor, tok["attention_mask"])  # (K, L)

        logger.debug(f"Tokenized input_ids shape: {input_ids_local.shape}, attention_mask shape: {attention_mask_local.shape}")

        # Step 2: Convert fixed tokens to embeddings using CLIP's token embedding layer
        with torch.no_grad():
            token_emb: torch.Tensor = clip_model.text_model.embeddings.token_embedding(
                input_ids_local.to(device=device)
            )  # (K, L, d)

        # Step 3: Extract prefix - only take the first token (BOS/start token)
        token_prefix: torch.Tensor = token_emb[:, :1, :]  # (K, 1, d)

        # Step 4: Extract suffix - all tokens after the first (includes class name and EOS)
        token_suffix: torch.Tensor = token_emb[:, 1:, :]  # (K, L-1, d)
        

        logger.debug(f"token_prefix shape: {token_prefix.shape}, token_suffix shape: {token_suffix.shape}")

        # Register fixed embeddings as buffers (non-trainable, moved to specified device)
        self.register_buffer("token_prefix", token_prefix.to(device))
        self.register_buffer("token_suffix", token_suffix.to(device))
        self.register_buffer("attention_mask", attention_mask_local.to(device))

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct prompt embeddings by concatenating prefix, learnable context, and suffix.
        
        Returns:
            prompt_embeds: (K, 1+n_ctx+L-1, d) - Full prompt embeddings for all classes
            attn_mask: (K, 1+n_ctx+L-1) - Corresponding attention mask
        """
        K: int = self.n_cls
        
        # Expand learnable context to all classes: (n_ctx, d) -> (K, n_ctx, d)
        ctx = self.ctx.unsqueeze(0).expand(K, -1, -1)
        
        # Concatenate: [BOS] + [learnable context] + [class name + EOS]
        prompt_embeds = torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)
        
        # Build attention mask: insert ones for the learnable context positions
        suffix_mask = self.attention_mask[:, 1:]  # (K, L-1) - mask for suffix tokens
        ctx_mask = torch.ones(
            K,
            self.n_ctx,
            device=self.device,
            dtype=suffix_mask.dtype,
        )  # (K, n_ctx) - all ones for learnable context
        prefix_mask = torch.ones(
            K,
            1,
            device=self.device,
            dtype=suffix_mask.dtype,
        )  # (K, 1) - one for BOS token

        # Concatenate masks in same order as embeddings
        attn_mask = torch.cat(
            [
                prefix_mask,
                ctx_mask,
                suffix_mask,
            ],
            dim=1,
        )  # (K, 1+n_ctx+L-1)
        
        logger.debug(f"PromptLearner forward: prompt_embeds shape={prompt_embeds.shape}, attn_mask shape={attn_mask.shape}")
        return prompt_embeds, attn_mask
    

class TextEncoder(nn.Module):
    """
    Text encoder wrapper that processes prompt embeddings through CLIP's text transformer.
    
    Takes pre-constructed prompt embeddings (from PromptLearner) and produces
    text features aligned with image features in CLIP's joint embedding space.
    """
    
    def __init__(
        self, device: torch.device = torch.device("cpu")
    ):
        super().__init__()
        # self.text_model = clip_model.text_model
        # self.text_projection = clip_model.text_projection
        self.device = device
        logger.info(f"TextEncoder initialized: device={device}")

    def forward(
        self,
        prompt_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        clip_model: CLIPModel,
    ) -> torch.Tensor:
        """
        Encode prompt embeddings to text features.
        
        Args:
            prompt_embeds: (K, Lp, d) - Prompt embeddings for K classes
            attention_mask: (K, Lp) - Attention mask for valid positions
            
        Returns:
            text_embeds: (K, projection_dim) - Text features in CLIP's joint space
        """
        K, Lp, _ = prompt_embeds.shape
        position_ids = torch.arange(Lp, device=prompt_embeds.device).unsqueeze(0)

        # Step 1: Add positional embeddings to prompt embeddings
        pos_emb = clip_model.text_model.embeddings.position_embedding(position_ids)  # (1, Lp, d)
        inputs_embeds = prompt_embeds + pos_emb  # (K, Lp, d)
        
        # Create causal attention mask for autoregressive text modeling
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape=(K, Lp),
            dtype=prompt_embeds.dtype,
            device=prompt_embeds.device,
        )

        # Prepare 4D attention mask if not using flash attention
        attn_mask_4d = None
        if attention_mask is not None and not getattr(clip_model.text_model, "_use_flash_attention_2", False):
            attn_mask_4d = _prepare_4d_attention_mask(attention_mask, prompt_embeds.dtype)

        # Step 2: Pass through transformer encoder
        out: BaseModelOutput = clip_model.text_model.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask_4d,
            causal_attention_mask=causal_attention_mask,
            output_attentions=False,
            output_hidden_states=False,
        )

        last_hidden = out.last_hidden_state  # (K, Lp, d)
        assert last_hidden is not None

        K = last_hidden.size(0)

        # Step 3: Apply final layer normalization
        last_hidden = clip_model.text_model.final_layer_norm(last_hidden)

        # Step 4: Extract EOS token hidden state as pooled output
        # EOS position is the last valid token (sum of attention mask - 1)
        eos_pos = attention_mask.sum(dim=-1) - 1  # (K,)
        pooled = last_hidden[torch.arange(K, device=self.device), eos_pos]  # (K, d)

        # Step 5: Project to CLIP's joint embedding space
        text_embeds = clip_model.text_projection(pooled)  # (K, projection_dim)
        
        logger.debug(f"TextEncoder forward: input shape=({K}, {Lp}), text_embeds shape={text_embeds.shape}")
        return text_embeds


class CustomCLIP(nn.Module):
    """
    Custom CLIP model with learnable text prompts for few-shot/zero-shot classification.
    
    This model freezes the original CLIP weights and only trains the learnable
    context tokens in the prompt, enabling efficient adaptation to new tasks.
    
    Architecture:
        - Frozen CLIP image encoder: Extracts image features
        - PromptLearner: Generates learnable prompt embeddings
        - TextEncoder: Encodes prompts to text features
        - Classification: Cosine similarity between image and text features
    """
    
    def __init__(
        self,
        clip_model: CLIPModel,
        processor: CLIPProcessor,
        classnames: list,
        n_ctx=32,
        template: str = "a photo of a {}",
        device=torch.device("cpu"),
    ):
        super().__init__()
        
        # Move CLIP model to device and freeze all parameters
        self.clip_model = clip_model.to(device)  # type: ignore
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.clip_model.eval()

        self.classnames = classnames
        # self.n_cls: int = len(classnames)
        self.n_ctx: int = n_ctx
        self.template = template
        self.processor = processor
        self.device = device
        
        # Initialize learnable prompt module
        self.prompt_learner = PromptLearner(
            clip_model=self.clip_model,
            processor=self.processor,
            classnames=self.classnames,
            n_ctx=self.n_ctx,
            template=self.template,
            device=self.device,
        )
        
        # Initialize text encoder wrapper
        self.text_encoder = TextEncoder(device=self.device)
        
        logger.info(f"CustomCLIP initialized: n_classes={len(classnames)}, n_ctx={n_ctx}, device={device}")
        logger.info(f"CustomCLIP classnames: {classnames}")

    def forward(self, pixel_values: torch.Tensor):
        """
        Forward pass for image classification.
        
        Args:
            pixel_values: (B, C, H, W) - Batch of input images
            
        Returns:
            logits: (B, K) - Classification logits for K classes
        """
        pixel_values = pixel_values.to(self.device)
        
        # Extract image features using frozen CLIP image encoder
        image_features = self.clip_model.get_image_features(pixel_values=pixel_values)  # type: ignore
        # L2 normalize image features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Generate text features from learnable prompts
        prompt_embeds, attn_mask = self.prompt_learner()
        text_features = self.text_encoder(prompt_embeds, attn_mask, self.clip_model)  # (K, projection_dim)
        # L2 normalize text features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarity scaled by learned temperature
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()  # (B, K)
        
        logger.debug(f"CustomCLIP forward: batch_size={pixel_values.shape[0]}, logits shape={logits.shape}, logit_scale={logit_scale.item():.4f}")
        return logits
    

    @torch.no_grad()
    def clone_prompt_only(self):
        new_model = CustomCLIP(
            clip_model=self.clip_model,
            processor=self.processor,
            classnames=self.classnames,
            n_ctx=self.n_ctx,
            template=self.template,
            device=self.device,
        )
        new_model.prompt_learner.load_state_dict(self.prompt_learner.state_dict(), strict=True)
        return new_model

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        new_obj = self.clone_prompt_only()
        memo[id(self)] = new_obj
        return new_obj