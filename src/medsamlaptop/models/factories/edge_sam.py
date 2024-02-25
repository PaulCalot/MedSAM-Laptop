# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from .interface import FactoryInterface

from ..products import MedEdgeSAM
from .. import core as core_models

prompt_embed_dim = 256
image_size = 1024
vit_patch_size = 16
image_embedding_size = image_size // vit_patch_size # 64

class MedEdgeSAMFactory(FactoryInterface):
    def __init__(self) -> None:
        super().__init__()

    # TODO: Add some params ?
    def create_model(self) -> MedEdgeSAM:
        image_encoder = core_models.RepViT(
            arch="m1",
            img_size=image_size,
            upsample_mode="bicubic"
        )

        prompt_encoder = core_models.PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16
        )

        mask_decoder = core_models.MaskDecoder(
            num_multimask_outputs=3,
                transformer=core_models.TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
        )

        model = MedEdgeSAM(
            image_encoder = image_encoder,
            mask_decoder = mask_decoder,
            prompt_encoder = prompt_encoder,
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
        )

        return model


