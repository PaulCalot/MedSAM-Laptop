from .interface import BaseTrainer

class EncoderDistiller(BaseTrainer):
    def handle_batch(self, batch):
        image = batch["image"].to(self.device)
        encoder_gts = batch["encoder_gts"].to(self.device)
        return (image,), encoder_gts

    def compute_loss(self, outputs, targets):
        loss = self.loss_fn(outputs, targets)
        return loss
