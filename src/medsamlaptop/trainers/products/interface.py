import abc
import torch
import numpy as np
import tqdm
import pathlib

from medsamlaptop.utils.checkpoint import Checkpoint
from medsamlaptop.plot.losses import plot_and_save_loss

class BaseTrainer(abc.ABC):
    def __init__(self
                 , model
                 , optimizer
                 , loss_fn
                 , lr_scheduler
                 , device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.device = device

    @abc.abstractmethod
    def handle_batch(self, batch):
        pass

    @abc.abstractmethod
    def compute_loss(self, outputs, targets):
        pass

    def train(self
              , train_loader
              , valid_loader
              , saving_dir: pathlib.Path
              , num_epochs: int
              , start_epoch: int =0
              , best_loss: float=1e10):
        train_losses = []
        eval_loss = best_loss
        for epoch in range(start_epoch + 1, num_epochs + 1):
            epoch_loss = []
            pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}")
            for batch in pbar:
                self.optimizer.zero_grad()
                data, targets = self.handle_batch(batch)
                outputs = self.model(*data)
                loss = self.compute_loss(outputs, targets)
                loss.backward()
                self.optimizer.step()

                # TODO: this takes time to compute this at each batch
                # roughly +50% compute time
                # do better
                loss_ = loss.item()
                epoch_loss.append(loss_)
                pbar.set_description(f"Epoch {epoch} Loss: {loss_:.4f} - Valid: {eval_loss:.4f} - Best: {best_loss:.4f}")

            # End of the epoch, compute validation loss
            with torch.no_grad():
                self.model.eval()
                eval_loss_list = []
                for batch in valid_loader:
                    data, targets = self.handle_batch(batch)
                    outputs = self.model(*data)
                    loss = self.compute_loss(outputs, targets)
                    eval_loss_list.append(loss.cpu())
                eval_loss = np.mean(eval_loss_list)
                self.model.train()

            avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
            train_losses.append(avg_epoch_loss)
            self.lr_scheduler.step(avg_epoch_loss)

            is_best = eval_loss < best_loss
            best_loss = min(eval_loss, best_loss)
            self._save_checkpoint(epoch, saving_dir, eval_loss, is_best)

        plot_and_save_loss(train_losses, saving_dir)

    def _save_checkpoint(self
                         , epoch
                         , saving_dir
                         , epoch_loss
                         , best_loss):
        checkpoint = Checkpoint(
            model_weights=self.model.state_dict()
            , epoch=epoch
            , optimizer_state=self.optimizer.state_dict()
            , loss=epoch_loss
            , best_loss=best_loss
        )
        checkpoint.save(saving_dir / "latest.pth")

        if epoch_loss < best_loss:
            print(f"New best loss: {best_loss:.4f} -> {epoch_loss:.4f}")
            best_loss = epoch_loss
            checkpoint.best_loss = best_loss
            checkpoint.save(saving_dir / "best.pth")
