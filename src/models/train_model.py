import logging
from pathlib import Path
from chess import WHITE
import click
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from sklearn.dummy import DummyClassifier
# from sklearn.linear_model import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.svm import SVC


def parse_fen(fen: str):
    def parseRow(fenRow: str):
        row = []
        for char in fenRow:
            if char.isdigit():
                row += ['blank'] * int(char)
            else:
                row.append(char)
        return row

    board = [parseRow(row) for row in fen.split("-")]
    return np.array(board)


BLACK_PIECES = ["p", "b", "k", "r", "n", "q"]
WHITE_PIECES = [x.upper() for x in BLACK_PIECES]


@click.command()
@click.argument("data_filepath", type=click.Path(exists=True))
def main(data_filepath: click.Path):
    path = Path(data_filepath)
    # LOAD DATA
    tr_X, tr_y = load_data(path.joinpath("train"))

    pipeline = Pipeline(
        [
            # ("pca", PCA(n_components=20)),
            # (
            #     "model",
            #     SVC(
            #         class_weight={np.nan: 1}
            #         | {x: 2 for x in BLACK_PIECES + WHITE_PIECES},
            #         max_iter=2e3,
            #     ),
            # ),
            ('model', DummyClassifier())
        ]
    )

    pipeline.fit(tr_X, tr_y)

    test_X, test_y = load_data(path.joinpath("test"))
    score = pipeline.score(test_X, test_y)
    print(score)

    ConfusionMatrixDisplay.from_estimator(pipeline, test_X, test_y)
    plt.savefig("./svc_pca_balanced.png")


def load_data(path: Path):
    X = []
    y = []
    for x in path.glob("*.npy"):
        X.append(np.load(x))
        y.append(parse_fen(x.stem))

    X = np.array(X)
    X = np.moveaxis(X, 0, -1)
    X = X.T.reshape(
        (
            X.shape[-1] * 8 * 8,  # number of squares
            50 * 50,  # pixels per square
        )
    )

    y = np.array(y).flatten()

    return X, y


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
