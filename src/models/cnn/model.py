import pytorch_lightning as pl
from torch import nn, optim

# NOTES
# Image: 400x400x3
# 8x8 Grid
#   => 50x50 cells
# 13 Classes (White/Black King/Queen/Rook/Bishop/Knight/Pawn + Blank Space)
class ChessCNN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.cnn_1 = nn.Conv2d(3, 5, (5, 5), 5) # Output: 5x80x80 (cells: 5x10x10)
        self.cnn_2 = nn.Conv2d(5, 8, (2, 2), 2) # Output: 8x40x40 (cells: 8x5x5)
        self.pool_1 = nn.MaxPool2d((5, 5), 5) # Output: 8x8x8 (cells: 8x1x1)
        self.cnn_3 = nn.Conv2d(8, 13, (1, 1)) # Output: 12x8x8 (cells: 13x1x1)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, X):
        X = self.cnn_1(X)
        X = self.cnn_2(X)
        X = self.pool_1(X)
        X = self.cnn_3(X)
        return X

    def training_step(self, batch, batch_idx):
        X, y = batch
        scores = self(X)
        loss = self.criterion(scores, y)
        self.log('tr_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        scores = self(X)
        loss = self.criterion(scores, y)
        
        pred = self.predict_step(batch)
        acc = (y == pred).float().mean()
        self.log_dict({
            'val_loss': loss,
            'val_acc': acc,
        })

    def test_step(self, batch, batch_idx):
        X, y = batch
        scores = self(X)
        loss = self.criterion(scores, y)
        
        pred = self.predict_step(batch)
        acc = (y == pred).float().mean()
        self.log_dict({
            'test_loss': loss,
            'test_acc': acc,
        })

    def predict_step(self, batch, batch_idx: int = None, dataloader_idx: int = 0):
        X, y = batch
        scores = self(X)
        return scores.argmax(dim=1)


    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=.02)