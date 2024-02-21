import torch
import tqdm
import time
import datetime

# model interface
from medsamlaptop.models.products import SegmentAnythingModelInterface
from medsamlaptop.utils.checkpoint import Checkpoint
from medsamlaptop.plot.losses import plot_and_save_loss

class MedSamFinetuner:
    def __init__(self
                 , model: SegmentAnythingModelInterface
                 , train_loader
                 , val_loader
                 , optimizer
                 , lr_scheduler
                 , loss_fn
                 , device) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.model.to(device)

    def train(self
              , saving_dir
              , num_epochs
              , start_epoch=0
              , best_loss=1e10):
        train_losses = []
        for epoch in range(start_epoch + 1, num_epochs):
            epoch_loss = [1e10 for _ in range(len(self.train_loader))]
            pbar = tqdm.tqdm(self.train_loader)
            for step, batch in enumerate(pbar):
                image = batch["image"]
                gt2D = batch["gt2D"]
                boxes = batch["bboxes"]
                self.optimizer.zero_grad()
                image, gt2D, boxes = image.to(self.device), gt2D.to(self.device), boxes.to(self.device)
                logits_pred, iou_pred = self.model(image, boxes)
                loss = self.loss_fn(
                    (logits_pred, iou_pred)
                    , gt2D
                )
                epoch_loss[step] = loss.item()
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                pbar.set_description(f"Epoch {epoch} at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, loss: {loss.item():.4f}")

            epoch_loss_reduced = sum(epoch_loss) / len(epoch_loss)
            train_losses.append(epoch_loss_reduced)
            self.lr_scheduler.step(epoch_loss_reduced)
            
            checkpoint = Checkpoint(
                self.model.state_dict()
                , epoch
                , self.optimizer.state_dict()
                , epoch_loss_reduced
                , best_loss
            )
            checkpoint.save(saving_dir / "latest.pth")

            if epoch_loss_reduced < best_loss:
                print(f"New best loss: {best_loss:.4f} -> {epoch_loss_reduced:.4f}")
                checkpoint.best_loss = epoch_loss_reduced
                checkpoint.save(saving_dir / "best.pth")

            epoch_loss_reduced = 1e10
        plot_and_save_loss(train_losses, saving_dir)

