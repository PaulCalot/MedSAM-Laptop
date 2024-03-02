import abc
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
        self.model = model
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
              , saving_dir: pathlib.Path
              , num_epochs: int
              , start_epoch: int =0
              , best_loss: float=1e10):
        train_losses = []
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

                epoch_loss.append(loss.item())
                pbar.set_description(f"Epoch {epoch} Loss: {loss.item():.4f}")

            avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
            train_losses.append(avg_epoch_loss)
            self.lr_scheduler.step(avg_epoch_loss)

            is_best = avg_epoch_loss < best_loss
            best_loss = min(avg_epoch_loss, best_loss)
            self._save_checkpoint(epoch, saving_dir, avg_epoch_loss, is_best)

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
