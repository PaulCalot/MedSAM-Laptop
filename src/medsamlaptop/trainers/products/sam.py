from .interface import BaseTrainer

class SamTrainer(BaseTrainer):
    def handle_batch(self, batch):
        # Extract and send data to the device (GPU/CPU)
        image = batch["image"].to(self.device)
        gt2D = batch["gt2D"].to(self.device)
        boxes = batch["bboxes"].to(self.device)
        return (image, boxes), gt2D  # Corresponding to inputs and targets respectively

    def compute_loss(self, outputs, targets):
        # Compute and return the loss
        logits_pred, iou_pred = outputs
        loss = self.loss_fn((logits_pred, iou_pred), targets)
        return loss
