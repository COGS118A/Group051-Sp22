from os import listdir, path
from pathlib import Path
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data.datapipes.utils.decoder import imagehandler
from torchdata import datapipes
from torchvision import transforms

BLACK_PIECES = ['p', 'b', 'k', 'r', 'n', 'q']
WHITE_PIECES = [x.upper() for x in BLACK_PIECES]
PIECE_ENCODING = {
    piece: i for (i, piece) in enumerate([np.nan] + BLACK_PIECES + WHITE_PIECES)
}

class ChessDataModule(pl.LightningDataModule):
    def __init__(self, path, batch_size = 128):
        super().__init__()
        self.path = path
        self.img_transforms = transforms.Compose([
            # transforms.Grayscale(),
            transforms.ToTensor()
        ])
        self.batch_size = batch_size


    def setup(self, stage) -> None:
        train_path = path.join(self.path, 'train')
        train_size = len(listdir(train_path))
        
        pipe = datapipes.iter.FileLister(train_path)
        pipe = datapipes.iter.Enumerator(pipe)
        pipe = datapipes.map.IterToMapConverter(pipe)

        val_size = len(pipe) // 4
        train_size = len(pipe) - val_size
        train_pipe, val_pipe = random_split(pipe, [train_size, val_size])
        
        train_pipe = datapipes.iter.IterableWrapper(train_pipe)
        train_pipe = datapipes.iter.FileOpener(train_pipe, 'b', length=train_size)
        train_pipe = datapipes.iter.RoutedDecoder(train_pipe, imagehandler('pil'))
        train_pipe = datapipes.iter.Mapper(train_pipe, self.transforms)
        self.train_pipe = train_pipe

        val_pipe = datapipes.iter.IterableWrapper(val_pipe)
        val_pipe = datapipes.iter.FileOpener(val_pipe, 'b', length=val_size)
        val_pipe = datapipes.iter.RoutedDecoder(val_pipe, imagehandler('pil'))
        val_pipe = datapipes.iter.Mapper(val_pipe, self.transforms)
        self.val_pipe = val_pipe

        test_pipe = datapipes.iter.FileLister(path.join(self.path, 'test'))
        test_pipe = datapipes.iter.FileOpener(test_pipe, 'b')
        test_pipe = datapipes.iter.RoutedDecoder(test_pipe, imagehandler('pil'))
        test_pipe = datapipes.iter.Mapper(test_pipe, self.transforms)
        self.test_pipe = test_pipe

    def parse_fen(self, fen: str):
        def parseRow(fenRow: str):
            row = []
            for char in fenRow:
                if char.isdigit():
                    row += [PIECE_ENCODING[np.nan]] * int(char)
                else:
                    row.append(PIECE_ENCODING[char])
            return row

        board = [parseRow(row) for row in fen.split('-')]
        return torch.tensor(board)

    def transforms(self, data):
        path, img = data
        fen = Path(path).stem
        board = self.parse_fen(fen)
        img = self.img_transforms(img)
        
        return img, board

    def train_dataloader(self):
        return DataLoader(self.train_pipe, batch_size=self.batch_size, num_workers=12)

    def val_dataloader(self):
        return DataLoader(self.val_pipe, batch_size=self.batch_size, num_workers=6)

    def test_dataloader(self):
        return DataLoader(self.test_pipe, batch_size=self.batch_size, num_workers=6)

