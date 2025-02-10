from functools import partial
import torch
import hashlib

from segment_anything.modeling import (
    ImageEncoderViT,
    PromptEncoder,
    TwoWayTransformer,
)
from roofsam.modeling.roofsam import RoofSam
from roofsam.modeling.class_decoder import ClassDecoder


# The actual expected SHA256 hash of the sam_vit_h_4b8939.pth file.
EXPECTED_SAM_VIT_H_CHECKPOINT_HASH = (
    "a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e"
)


def compute_file_hash(filepath, hash_algo="sha256"):
    """
    Computes the hash of a file using the specified hash algorithm.
    """
    hash_func = hashlib.new(hash_algo)
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(4096), b""):
            hash_func.update(block)
    return hash_func.hexdigest()


def build_roofsam_from_sam_vit_h_checkpoint(
    num_classes, sam_checkpoint, roof_sam_mask_decoder_checkpoint=None
):
    # Compute and check the hash for the sam_checkpoint file.
    checkpoint_hash = compute_file_hash(sam_checkpoint)
    if checkpoint_hash != EXPECTED_SAM_VIT_H_CHECKPOINT_HASH:
        raise ValueError(
            f"Invalid sam_checkpoint hash: {checkpoint_hash}. "
            f"Expected: {EXPECTED_SAM_VIT_H_CHECKPOINT_HASH}. "
            f"Make sure to load the sam_vit_h_4b8939.pth checkpoint."
        )

    # Build the RoofSAM model using the provided sam_checkpoint.
    roofsam = _build_roofsam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        num_classes=num_classes,
        sam_checkpoint=sam_checkpoint,
    )

    if roof_sam_mask_decoder_checkpoint is not None:
        state_dict = torch.load(roof_sam_mask_decoder_checkpoint)
        roofsam.mask_decoder.load_state_dict(state_dict)
    else:
        print("Returning RoofSAM with uninitialized class prediction head!")

    return roofsam


def _build_roofsam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    num_classes,
    sam_checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    roofsam = RoofSam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=ClassDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            num_classes=num_classes,
            mlp_hidden_dim=prompt_embed_dim,
            mlp_layers=3,
        ),
    )

    if sam_checkpoint is not None:
        with open(sam_checkpoint, "rb") as f:
            state_dict = torch.load(f)
        missing_keys, unexpected_keys = roofsam.load_state_dict(
            state_dict, strict=False
        )
        expected_missing = [
            x.startswith("mask_decoder.class_pred_mlp") for x in missing_keys
        ]
        if not all(expected_missing) or unexpected_keys:
            raise ValueError("SAM state dict doesn't contain the expected keys")

    return roofsam
