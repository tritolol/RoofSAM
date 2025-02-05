from segment_anything.segment_anything.modeling.mask_decoder import MaskDecoder
import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Tuple, Type


class ClassDecoder(MaskDecoder):
    def __init__(self, 
                 *, 
                 transformer_dim, 
                 transformer, 
                 num_classes,
                 mlp_hidden_dim,
                 mlp_layers,
                 num_multimask_outputs = 3, 
                 activation = nn.GELU, 
                 iou_head_depth = 3, 
                 iou_head_hidden_dim = 256,
                 ) -> None:
        super().__init__(transformer_dim=transformer_dim,
                         transformer=transformer,
                         num_multimask_outputs=num_multimask_outputs, 
                         activation=activation, 
                         iou_head_depth=iou_head_depth, 
                         iou_head_hidden_dim=iou_head_hidden_dim
                         )
        self.class_pred_mlp = MLP(transformer_dim, mlp_hidden_dim, num_classes, mlp_layers, softmax_output=False)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs

        Returns:
          torch.Tensor: batched predicted class probabilitites
        """
        probs = self.predict_classes(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        return probs

    def predict_classes(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        # chanded: expect the caller to do this
        # src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = image_embeddings + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        # iou_token_out = hs[:, 0, :]
        # mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        prompt_tokens_out = hs[:, (1 + self.num_mask_tokens) : -1, :]   # exclude padding token

        logits = self.class_pred_mlp(prompt_tokens_out).permute((0,2,1))    # (B, C, N_points)

        return logits
    
# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        softmax_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.softmax_output = softmax_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.softmax_output:
            x = F.softmax(x, dim=-1)
        return x