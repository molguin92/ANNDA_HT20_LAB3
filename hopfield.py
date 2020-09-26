import itertools
import warnings
from typing import Literal, Optional

import numpy as np
from numpy.random import default_rng

rand_gen = default_rng()


class ConvergenceWarning(Warning):
    pass


class HopfieldNetwork:
    """
    Implementation of a Hpofield recurrent neural network.
    """

    def __init__(self):
        super(HopfieldNetwork, self).__init__()
        self._w = np.empty(0)

    def train(self, X: np.ndarray, convergence_threshold: int = 5) -> None:
        """
        Trains the network on a set of input patters.

        Input patterns should be provided as a matrix X of dimensions MxN,
        where M corresponds to the number of patterns to learn and N to the
        number of attributes per pattern.

        This method automatically sets up the number of nodes in this network to
        match the dimensions of the input patterns.

        :param X: A matrix of patterns to learn.
        :param convergence_threshold: Number of iterations the weight matrix
        needs to be constant for this method to consider it to have converged.
        :return:
        """

        # X has dims samples x attrs
        # iterate until convergence

        X = X.copy()
        attr_dims = X.shape[1]

        self._w = np.zeros(shape=(attr_dims, attr_dims))
        convergence_count = 0
        epochs = 0

        while True:
            previous_w = self._w.copy()
            for pattern in X:
                w = np.outer(pattern.T, pattern)
                # np.fill_diagonal(w, 0)
                self._w += w

            np.fill_diagonal(self._w, 0)
            self._w /= attr_dims
            epochs += 1

            convergence_count = convergence_count + 1 \
                if np.all(np.isclose(self._w, previous_w)) else 0

            if convergence_count >= convergence_threshold:
                break

    def _synchronous_recall(self, y: np.ndarray) -> np.ndarray:
        new_y = np.dot(self._w, y)
        new_y[new_y >= 0] = 1  # x >= 0 -> 1
        new_y[new_y < 0] = -1  # x < 0 -> -1
        return new_y

    def _asynchronous_recall(self, y: np.ndarray) -> np.ndarray:
        new_y = y.copy()
        indices = rand_gen.permutation(new_y.size)

        for i in indices:
            # iteratively calculate new updates
            new_i = np.sum(np.multiply(self._w[i, :], new_y))
            new_y[i] = -1 if new_i < 0 else 1

        return new_y

    def recall(self, Xd: np.ndarray,
               mode: Literal['asynchronous', 'synchronous'] = 'asynchronous',
               convergence_threshold: int = 5,
               max_iter: Optional[int] = None) -> np.ndarray:
        """
        Tries to update a set of given input patterns to match the stored 
        patterns in this network.
        
        :param Xd: Matrix of input patterns to use as inputs to the recall.
        :param mode: 'asynchronous' or 'synchronous'. Asynchronous recall
        updates every unit in the patters one at the time, synchronous recall
        updates the whole pattern at once.
        :param convergence_threshold: Number of iterations the output pattern
        needs to be constant for this method to consider it to have converged.
        :param max_iter: Maximum iterations before this method gives up on
        finding a convergence.
        :return: A matrix of the same dimensions as the input matrix
        containing the recalled patterns.
        """

        max_iter = 10 * np.log(self._w.shape[0]) \
            if max_iter is None else max_iter

        if mode == 'asynchronous':
            recall_fn = self._asynchronous_recall
        elif mode == 'synchronous':
            recall_fn = self._synchronous_recall
        else:
            raise RuntimeError(f'Invalid mode: {mode}!')

        Y = np.empty(shape=Xd.shape)

        for i, pattern in enumerate(Xd):
            new_y = pattern.copy()
            convergence_count = 0
            iterations = 0

            while True:
                prev_y = new_y.copy()
                new_y = recall_fn(prev_y)

                convergence_count = convergence_count + 1 \
                    if np.all(np.isclose(new_y, prev_y)) else 0

                iterations += 1

                if convergence_count >= convergence_threshold:
                    break
                elif iterations >= max_iter:
                    warnings.warn(f'Pattern did not converge after '
                                  f'the maximum number of iterations ('
                                  f'{int(np.floor(max_iter)):d})!',
                                  ConvergenceWarning)
                    break

            Y[i, :] = new_y

        return Y


if __name__ == '__main__':
    # note that x3 is not orthogonal with respect to either x1 or x2,
    # and thus causes crosstalk in the network!
    # that's why we can't get perfect recall if we train the network with all
    # three vectors
    # if we ignore x3, x1d and x2d converge to x1 and x2.

    x1 = [-1, -1, 1, -1, 1, -1, -1, 1]
    x2 = [-1, -1, -1, -1, -1, 1, -1, -1]
    x3 = [-1, 1, 1, -1, -1, 1, -1, 1]

    X = np.array([x1, x2, x3])

    for (i, x), (j, y) in itertools.product(enumerate(X), enumerate(X)):
        if i == j:
            continue
        elif not np.isclose(np.dot(x, y), 0):
            print(f'x{i + 1} and x{j + 1} are not orthogonal!')

    # make sure the patterns are fixed points
    nn = HopfieldNetwork()
    nn.train(X)

    assert np.all(nn.recall(X) == X)

    x1d = [1, -1, 1, -1, 1, -1, -1, 1]
    x2d = [1, 1, -1, -1, -1, 1, -1, -1]
    x3d = [1, 1, 1, -1, 1, 1, -1, 1]
    Xd = np.array([x1d, x2d, x3d])

    Xp = nn.recall(Xd)

    print(Xp == X)
