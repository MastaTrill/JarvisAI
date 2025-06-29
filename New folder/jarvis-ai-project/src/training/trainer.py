class Trainer:
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader, device):
        """
        Initializes the Trainer class.

        Parameters:
        model: The model to be trained.
        optimizer: The optimizer for updating model weights.
        loss_fn: The loss function for training.
        train_loader: DataLoader for the training dataset.
        val_loader: DataLoader for the validation dataset.
        device: The device to run the training on (CPU or GPU).
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

    def train_one_epoch(self):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        total_loss = 0

        for batch in self.train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(self.train_loader)
        return average_loss

    def validate(self):
        """
        Validates the model on the validation dataset.
        """
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in self.val_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()

        average_loss = total_loss / len(self.val_loader)
        return average_loss

    def train(self, num_epochs):
        """
        Trains the model for a specified number of epochs.

        Parameters:
        num_epochs: The number of epochs to train the model.
        """
        for epoch in range(num_epochs):
            train_loss = self.train_one_epoch()
            val_loss = self.validate()
            print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')