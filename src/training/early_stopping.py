class EarlyStopping:

    def __init__(self, patience=5):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, validation_loss):
        if validation_loss < self.best_loss:
            self.best_loss = validation_loss
            self.counter = 0
            return False
        self.counter += 1

        return self.counter >= self.patience