from .interface import BaseTrainer

class SamTrainer(BaseTrainer):
    def handle_batch(self, batch):
        image = batch["image"].to(self.device)
        gt2D = batch["gt2D"].to(self.device)
        boxes = batch["bboxes"].to(self.device)
        return (image, boxes), gt2D

    def compute_loss(self, outputs, targets):
        logits_pred, iou_pred = outputs
        loss = self.loss_fn((logits_pred, iou_pred), targets)
        return loss
