import logging
from pathlib import Path
import click
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import numpy as np
from tqdm import tqdm
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


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
    test_X, test_y = load_data(path.joinpath('test'))


    '''

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(tr_X, tr_y)

    ConfusionMatrixDisplay.from_estimator(knn, test_X, test_y)
    plt.savefig("./knn.png")

    '''

    '''
    print(knn.classes_)
    print(knn.score(test_X, test_y))
    predicted = knn.predict(test_X)

    uniques, counts = np.unique(predicted, return_counts=True)
    for i, unique in enumerate(uniques):
        print(unique + ": " + str(counts[i]))

    uniques, counts = np.unique(test_y, return_counts=True)
    for i, unique in enumerate(uniques):
        print(unique + ": " + str(counts[i]))
    '''

    knn = KNeighborsClassifier()
    
    parameters = {
        'n_neighbors':range(8, 20),
        'weights': ('uniform', 'distance'),
    }

    grid_search = GridSearchCV(knn, parameters, verbose=4, cv=ShuffleSplit(n_splits=1, test_size=0.3))
    best_model = grid_search.fit(tr_X, tr_y)

    
    print(grid_search.best_estimator_.get_params())
    print(best_model.cv_results_)
    


def load_data(path: Path):
    X = []
    y = []
    for x in tqdm(path.glob("*.npy")):
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
