"""
Derived from segment_anything/modeling/sam.py
"""

from typing import Any, Dict
import torch
from torch import nn
from segment_anything.modeling import ImageEncoderViT
from segment_anything.modeling import PromptEncoder
from roofsam.modeling.class_decoder import ClassDecoder


class RoofSam(nn.Module):
    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: ClassDecoder,
    ) -> None:
        """
        RoofSAM classifies input prompts based on an image.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (ClassDecoder): Predicts classes from the image embeddings
            and encoded prompts. The name is kept from SAM for parameter loading
            compatibility.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder

    def forward(
        self,
        batched_input: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (dict): A dict containing different batched input elements.
            It is expected to contain the following keys:
              "embeddings": (torch.Tensor) image embeddings from the image encoder,
              "point_coords": (torch.Tensor) point coordinates in image coordinates
              with shape BxNx2,
              "point_labels": (torch.Tensor) The point labels (positive or negative)
                for each point coordinate with shape BxN,

        Returns:
          logits: (torch.Tensor) Batched class probability logits with
            shape BxNxC
        """
        image_embeddings = batched_input["embeddings"]
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=(batched_input["point_coords"], batched_input["point_labels"]),
            boxes=None,  # never use boxes or masks
            masks=None,
        )

        logits = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
        )

        return logits
