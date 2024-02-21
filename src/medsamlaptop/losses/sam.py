import torch
import monai
import os

class SAMLoss:
    # TODO: make it compatible with pytorch loss
    def __init__(self
                 , seg_loss_weight: float
                 , ce_loss_weight: float
                 , iou_loss_weight: float) -> None:
        self.seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
        self.ce_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.iou_loss = torch.nn.MSELoss(reduction='mean')
        self.seg_loss_weight = seg_loss_weight
        self.ce_loss_weight = ce_loss_weight
        self.iou_loss_weight = iou_loss_weight

    # TODO: make pred an object containing what is needed
    # same for truth
    def __call__(self, pred, truth) -> os.Any:
        logits_pred, iou_pred = pred
        gt2D = truth
        
        l_seg = self.seg_loss(logits_pred, gt2D)
        l_ce = self.ce_loss(logits_pred, gt2D.float())
        iou_gt = IoULoss(torch.sigmoid(logits_pred) > 0.5, gt2D.bool())
        l_iou = self.iou_loss(iou_pred, iou_gt)

        mask_loss = self.seg_loss_weight * l_seg + self.ce_loss_weight * l_ce
        loss = mask_loss + self.iou_loss_weight * l_iou
        return loss
    
def IoULoss(result, reference):    
    intersection = torch.count_nonzero(torch.logical_and(result, reference), dim=[i for i in range(1, result.ndim)])
    union = torch.count_nonzero(torch.logical_or(result, reference), dim=[i for i in range(1, result.ndim)])
    iou = intersection.float() / union.float()
    return iou.unsqueeze(1)