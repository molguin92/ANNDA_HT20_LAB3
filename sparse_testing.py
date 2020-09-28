import itertools
from typing import Tuple

import multiprocess
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.special
from numpy.random import default_rng
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from hopfield import HopfieldNetwork

rand_gen = default_rng()


# sparse patterns
def generate_sparse_patterns(xdims: int,
                             activity: float,
                             max_patterns: int = 50) -> np.ndarray:
    # set a limit for how many patterns we grab based on calculating the
    # possible k-combinations (with replacement) of indices
    # in each sample based on the activity

    ones_per_sample = int(np.floor(activity * xdims))
    n_patterns = scipy.special.comb(xdims, ones_per_sample, exact=True)
    n_patterns = min(max_patterns, n_patterns)

    X = scipy.sparse.random(n_patterns, xdims, density=activity,
                            random_state=rand_gen, dtype='int32').A
    X[X != 0] = 1
    X_unique = np.unique(X, axis=0)
    while X_unique.shape[0] != n_patterns:
        missing = n_patterns - X_unique.shape[0]
        X_missing = scipy.sparse.random(missing, xdims, density=activity,
                                        random_state=rand_gen, dtype='int32').A
        X_missing[X_missing != 0] = 1
        X_unique = np.unique(np.append(X_unique, X_missing, axis=0), axis=0)

    return X_unique


def _mse_recall_sparse_hopfield(arg_tuple: Tuple) -> float:
    X, theta = arg_tuple
    nn = HopfieldNetwork()
    nn.train(X, sparse=True)

    Xp = nn.recall(X, sparse=True, sparse_theta=theta)

    return mean_squared_error(X, Xp)


def test_sparse(activity: float, xdims: int = 1024) -> pd.DataFrame:
    results = []
    X = generate_sparse_patterns(xdims, activity)

    with multiprocess.Pool() as pool:
        for theta in np.power(10.0, np.arange(-1, 6)):
            # generate slices of patterns
            n_patterns = list(range(1, X.shape[0]))
            patterns = [X[:i, :] for i in n_patterns]

            mses = tqdm(
                pool.imap(_mse_recall_sparse_hopfield,
                          zip(patterns, itertools.repeat(theta))),
                total=len(patterns),
                desc=f'Testing sparse recall | '
                     f'Activity = {activity} | '
                     f'Theta = {theta}',
                leave=False
            )

            results.extend(
                [
                    {
                        'activity'  : activity,
                        'theta'     : theta,
                        'n_patterns': i,
                        'recall_mse': error
                    } for i, error in zip(n_patterns, mses)
                ]
            )

    return pd.DataFrame(results)


if __name__ == '__main__':
    activities = [0.01, 0.05, 0.1]
    pd.concat([test_sparse(a) for a in activities]).to_csv('sparse_tests.csv',
                                                           index=False)
