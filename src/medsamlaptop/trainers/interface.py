import abc
import torch

class TrainerInterface(abc.ABC):
    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, device='cpu'):
        """
        Initializes the Trainer interface with common elements needed for training.

        Args:
            model: The PyTorch model to be trained.
            train_loader: DataLoader for the training dataset.
            val_loader: DataLoader for the validation dataset, can be None.
            optimizer: Optimizer used for training.
            loss_fn: Loss function used for training.
            device: The device to run the training on ('cpu' or 'cuda').
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    @abc.abstractmethod
    def train(self, epochs):
        """
        Abstract method for training the model.

        Args:
            epochs: Number of epochs to train the model.
        """
        pass

    @abc.abstractmethod
    def validate(self):
        """
        Abstract method for validating the model.
        """
        pass
