import itertools
from typing import Tuple

import multiprocess
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.special
from numpy.random import default_rng
from tqdm import tqdm

from hopfield import HopfieldNetwork

rand_gen = default_rng()

def gen_sparse_patterns(xdims: int,
                        activity: float)


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


def _error_recall_sparse_hopfield(arg_tuple: Tuple) -> float:
    X, theta = arg_tuple
    nn = HopfieldNetwork()
    nn.train(X, sparse=True)

    Xp = nn.recall(X, sparse=True, sparse_theta=theta)

    # fraction of patters which were not remembered
    error = np.sum(np.any(X != Xp, axis=1)) / Xp.shape[0]
    return error


def test_sparse(activity: float, xdims: int = 100) -> pd.DataFrame:
    results = []
    max_pats = int(np.floor(1.5 * xdims * 0.138))
    X = generate_sparse_patterns(xdims, activity, max_patterns=max_pats)

    with multiprocess.Pool() as pool:
        # for theta in np.power(10.0, np.arange(-5, 1)):
        for theta in np.power(10.0, np.linspace(-3, 0, 7)):
            # generate slices of patterns
            n_patterns = list(range(1, X.shape[0]))

            patterns = [
                X[rand_gen.choice(np.arange(X.shape[0]), size=i), :].copy()
                for i in n_patterns
            ]

            errors = tqdm(
                pool.imap(_error_recall_sparse_hopfield,
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
                        'activity'    : activity,
                        'theta'       : theta,
                        'n_patterns'  : i,
                        'recall_error': error
                    } for i, error in zip(n_patterns, errors)
                ]
            )

    return pd.DataFrame(results)


if __name__ == '__main__':
    activities = [0.01, 0.05, 0.1]
    pd.concat([test_sparse(a) for a in activities]).to_csv('sparse_tests.csv',
                                                           index=False)

    # X = generate_sparse_patterns(100, activity=0.1, max_patterns=10)
    # X_nz = np.flatnonzero(X)
    #
    # nn = HopfieldNetwork()
    # nn.train(X, sparse=True)
    #
    # Xp = nn.recall(X, sparse=True, sparse_theta=0.035)
    # Xp_nz = np.flatnonzero(Xp)
    #
    # print(X == Xp)
    # print(np.any(X != Xp, axis=1))
    # print(np.sum(np.any(X != Xp, axis=1)) / Xp.shape[0])
