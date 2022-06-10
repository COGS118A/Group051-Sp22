import logging
from pathlib import Path
import click
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, ShuffleSplit, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import numpy as np
from tqdm import tqdm
import pickle
from os.path import exists
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

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



    
    dt = DecisionTreeClassifier(
        criterion='gini',
        splitter='best',
        max_depth=5,
        #class_weight='balanced'
    )


    # Delete dtmodel to refresh model.
    if exists("dtmodel"):
        dt =pickle.load(open('dtmodel', 'rb'))
        print("Restored DT from file")
    else:
        dt.fit(tr_X, tr_y, )
        pickle.dump(dt, open("dtmodel", "wb"))
    
    
    print("Fit Complete")
    

    print(dt.classes_)
    print(dt.score(test_X, test_y))

    #ConfusionMatrixDisplay.from_estimator(dt, test_X, test_y)
    #plt.savefig("./decisiontreewithoutclassweight.png")



    '''


    pipeline = Pipeline([
        ("pca", PCA()),
        ("model", DecisionTreeClassifier())
    ])

    '''
    
    '''
    parameters = {
        'model__criterion': ['gini'],
        'model__splitter': ['best'],
        'model__max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10],
        'model__min_samples_split': [2, 3, 4],
        'model__min_samples_leaf': [5.0, 10.0, 20.0],
        'model__max_features': ['sqrt', 'log2'],
        'pca__n_components': [30, 50, 80],
        'model__class_weight': ['balanced']
    }

    
    parameters = {
        'model__criterion': ['gini'],
        'model__splitter': ['best'],
        'model__max_depth': [2],
        'model__min_samples_split': [2],
        'model__min_samples_leaf': [5.0],
        #'model__max_features': ['sqrt', 'log2'],
        #'pca__n_components': [30, 50, 80],
        'model__class_weight': ['balanced']
    }

    

    grid_search = GridSearchCV(pipeline, parameters, verbose=4, cv=ShuffleSplit(n_splits=1, test_size=0.3))
    # grid_search = GridSearchCV(pipeline, parameters, verbose=4, cv=StratifiedKFold())
    best_model = grid_search.fit(tr_X, tr_y)

    
    print(grid_search.best_estimator_.get_params())
    print(best_model.cv_results_)

    '''



    #results = grid_search.predict(test_X)
    #print(np.unique(results))


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
