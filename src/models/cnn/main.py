import click
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from model import ChessCNN
from data import ChessDataModule


@click.command()
@click.argument("data", type=click.Path(exists=True))
def main(data: str):
    dataset = ChessDataModule(data)
    model = ChessCNN()

    logger = TensorBoardLogger("tb_logs", name="chess-cnn")
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
    )
    trainer.fit(model, datamodule=dataset)
    trainer.test(model, datamodule=dataset)


if __name__ == "__main__":
    main()
