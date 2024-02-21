# from medsamlaptop.trainers.
# import torch


# class Trainer:
#     def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, device='cpu'):
#         """
#         Initializes the Trainer class.

#         Args:
#             model: The PyTorch model to be trained.
#             train_loader: DataLoader for the training dataset.
#             val_loader: DataLoader for the validation dataset.
#             optimizer: Optimizer used for training.
#             loss_fn: Loss function used for training.
#             device: The device to run the training on ('cpu' or 'cuda').
#         """
#         self.model = model
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.optimizer = optimizer
#         self.loss_fn = loss_fn
#         self.device = device
#         self.model.to(device)  # Move the model to the specified device

#     def train(self, epochs):
#         """
#         Trains the model for a specified number of epochs.

#         Args:
#             epochs: Number of epochs to train the model.
#         """
#         self.model.train()  # Set the model to training mode
#         for epoch in range(epochs):
#             running_loss = 0.0
#             for batch_idx, (data, targets) in enumerate(self.train_loader):
#                 # Move data to the specified device
#                 data, targets = data.to(self.device), targets.to(self.device)

#                 # Forward pass
#                 outputs = self.model(data)
#                 loss = self.loss_fn(outputs, targets)

#                 # Backward pass and optimize
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()

#                 running_loss += loss.item()

#                 # Print statistics or save logs/checkpoints if necessary
#                 if batch_idx % 100 == 99:  # print every 100 mini-batches
#                     print(f'Epoch: {epoch + 1}, Batch: {batch_idx + 1}, Loss: {running_loss / 100}')
#                     running_loss = 0.0

#             # Validation step after each epoch, if validation loader is provided
#             if self.val_loader:
#                 self.validate()

#     def validate(self):
#         """
#         Validates the model performance on the validation dataset.
#         """
#         self.model.eval()  # Set the model to evaluation mode
#         total_loss = 0.0
#         with torch.no_grad():  # No need to track gradients during validation
#             for data, targets in self.val_loader:
#                 data, targets = data.to(self.device), targets.to(self.device)
#                 outputs = self.model(data)
#                 loss = self.loss_fn(outputs, targets)
#                 total_loss += loss.item()

#         average_loss = total_loss / len(self.val_loader)
#         print(f'Validation Loss: {average_loss}')
