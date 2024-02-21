from .interface import FactoryInterface

from ..products import MedSAMLite
from .. import core as core_models

class MedSAMLiteFactory(FactoryInterface):
    def __init__(self) -> None:
        super().__init__()

    # TODO: Add some params ?
    def create_model(self) -> MedSAMLite:
        medsam_lite_image_encoder = core_models.TinyViT(
            img_size=256,
            in_chans=3,
            embed_dims=[
                64, ## (64, 256, 256)
                128, ## (128, 128, 128)
                160, ## (160, 64, 64)
                320 ## (320, 64, 64) 
            ],
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 5, 10],
            window_sizes=[7, 7, 14, 7],
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.0,
            use_checkpoint=False,
            mbconv_expand_ratio=4.0,
            local_conv_size=3,
            layer_lr_decay=0.8
        )

        medsam_lite_prompt_encoder = core_models.PromptEncoder(
            embed_dim=256,
            image_embedding_size=(64, 64),
            input_image_size=(256, 256),
            mask_in_chans=16
        )

        medsam_lite_mask_decoder = core_models.MaskDecoder(
            num_multimask_outputs=3,
                transformer=core_models.TwoWayTransformer(
                    depth=2,
                    embedding_dim=256,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=256,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
        )

        medsam_lite_model = MedSAMLite(
            image_encoder = medsam_lite_image_encoder,
            mask_decoder = medsam_lite_mask_decoder,
            prompt_encoder = medsam_lite_prompt_encoder
        )

        return medsam_lite_model