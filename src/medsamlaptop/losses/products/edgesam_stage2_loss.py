import torch
import monai

class EdgeSamStage2Loss(torch.nn.Module):
    def __init__(self
                 , seg_loss_weight: float
                 , ce_loss_weight: float) -> None:
        super(EdgeSamStage2Loss, self).__init__()
        self.seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
        self.ce_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.seg_loss_weight = seg_loss_weight
        self.ce_loss_weight = ce_loss_weight

    def forward(self, pred, truth):
        # binary thresholding
        gt2D = truth > 0.5
        l_seg = self.seg_loss(pred, gt2D)
        l_ce = self.ce_loss(pred, gt2D.float())
        mask_loss = self.seg_loss_weight * l_seg + self.ce_loss_weight * l_ce
        return mask_loss
