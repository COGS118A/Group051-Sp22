import logging
from pathlib import Path
import click
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import numpy as np

def read_fen(fen: str):
    board = []
    for char in fen:
        if(char.isdigit()):
            board += [np.nan] * int(char)
        elif(char != '-'):
            board += char
    board = np.reshape(board, (8, 8))
    return board

@click.command()
@click.argument("data_filepath", type=click.Path(exists=True))
def main(data_filepath: click.Path):
    path = Path(data_filepath)
    # LOAD DATA
    tr_X, tr_y = load_data(path.joinpath('train'))

    pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=60)),
        ("logistic", LogisticRegression(max_iter=200))
    ])
    
    params = {
    "logistic__solver" : ['saga'],
    "logistic__penalty" : ['l2', 'l1'],
    "logistic__C" : [0.01, 0.1, 1, 10],
    "logistic__class_weight" : ['balanced']
    }

    gscv = GridSearchCV(pipeline, cv=3, scoring='accuracy',param_grid=params, verbose=3)
    gscv.fit(tr_X, tr_y)

    test_X, test_y = load_data(path.joinpath('test'))
    score = gscv.score(test_X, test_y)
    print(score)
    print(gscv.best_params_)
    print(gscv.cv_results_)


def load_data(path: Path):
    X = []
    y = []
    for x in path.glob("*.npy"):
        X.append(np.load(x))
        y.append(read_fen(x.stem))

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

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
