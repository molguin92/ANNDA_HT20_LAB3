import itertools
import warnings
from typing import Any, Callable, Literal, Optional

import numpy as np
from numpy.random import default_rng

rand_gen = default_rng()


class ConvergenceWarning(Warning):
    pass


class HopfieldNetwork:
    """
    Implementation of a Hopfield recurrent neural network.
    """

    def __init__(self):
        super(HopfieldNetwork, self).__init__()
        self._w = np.empty(0)

    def train(self, X: np.ndarray,
              self_connections: bool = False,
              sparse: bool = False,
              convergence_threshold: int = 5) -> None:
        """
        Trains the network on a set of input patters.

        Input patterns should be provided as a matrix X of dimensions MxN,
        where M corresponds to the number of patterns to learn and N to the
        number of attributes per pattern.

        This method automatically sets up the number of nodes in this network to
        match the dimensions of the input patterns.

        :param X: A matrix of patterns to learn.
        :param self_connections: Allow self connections in the weight matrix
        (i.e. if false, this will mean that the weight matrix will have a
        fully-zeroed diagonal).
        :param convergence_threshold: Number of iterations the weight matrix
        needs to be constant for this method to consider it to have converged.
        :return:
        """

        # X has dims samples x attrs
        # iterate until convergence

        X = X.copy()
        attr_dims = X.shape[1]

        # check values in matrices match
        assert np.all(np.unique(X) == [0, 1]) \
            if sparse else np.all(np.unique(X) == [-1, 1])

        self._w = np.zeros(shape=(attr_dims, attr_dims))
        convergence_count = 0
        epochs = 0

        # for sparse patterns
        rho = np.sum(X) / X.size if sparse else 0

        while True:
            previous_w = self._w.copy()
            for pattern in X:
                w = np.outer(pattern.T - rho, pattern - rho)
                # np.fill_diagonal(w, 0)
                self._w += w

            if not self_connections:
                np.fill_diagonal(self._w, 0)
            self._w /= attr_dims
            epochs += 1

            convergence_count = convergence_count + 1 \
                if np.all(np.isclose(self._w, previous_w)) else 0

            if convergence_count >= convergence_threshold:
                break

    def _synchronous_recall(self,
                            y: np.ndarray,
                            sparse: bool,
                            theta: float,
                            callback: Callable[[int, np.ndarray], Any]) \
            -> np.ndarray:
        new_y = np.dot(self._w, y) - theta
        new_y[new_y >= 0] = 1  # x >= 0 -> 1
        new_y[new_y < 0] = -1  # x < 0 -> -1

        new_y = 0.5 + (0.5 * new_y) if sparse else new_y

        callback(-1, new_y)
        return new_y

    def _asynchronous_recall(self, y: np.ndarray,
                             sparse: bool,
                             theta: float,
                             random_units: bool,
                             callback: Callable[[int, np.ndarray], Any]) \
            -> np.ndarray:
        new_y = y.copy()
        indices = rand_gen.permutation(new_y.size) \
            if random_units else np.arange(new_y.size)

        for i in indices:
            # iteratively calculate new updates
            new_i = np.sum(np.multiply(self._w[i, :], new_y)) - theta
            new_i = -1 if new_i < 0 else 1
            new_y[i] = 0.5 + (0.5 * new_i) if sparse else new_i
            callback(i, new_y)

        return new_y

    def recall(self, Xd: np.ndarray,
               mode: Literal['asynchronous', 'synchronous'] = 'asynchronous',
               random_units: bool = False,
               convergence_threshold: int = 5,
               max_iter: Optional[int] = None,
               sparse: bool = False,
               sparse_theta: float = 0.1,
               callback: Callable[[int, int, int, np.ndarray], Any] =
               lambda epoch, p_index, unit, pattern: None) -> np.ndarray:
        """
        Tries to update a set of given input patterns to match the stored
        patterns in this network.

        :param Xd: Matrix of input patterns to use as inputs to the recall.
        :param mode: 'asynchronous' or 'synchronous'. Asynchronous recall
        updates every unit in the patters one at the time, synchronous recall
        updates the whole pattern at once.
        :param random_units: When performing asynchronous recall, update
        units in random order.
        :param convergence_threshold: Number of iterations the output pattern
        needs to be constant for this method to consider it to have converged.
        :param max_iter: Maximum iterations before this method gives up on
        finding a convergence.
        :param sparse: Recall in sparse mode.
        :param sparse_theta: Threshold to use in sparse mode. Ignored otherwise.
        :param callback: A function to be executed periodically during the
        recall procedure. This function should take four parameters: an int
        representing the current epoch, an int representing the index of the 
        current pattern, an int corresponding  to the last updated unit (will 
        always be -1 in synchronous mode), and a np.ndarray containing the 
        current state of the pattern. This function will be executed for each 
        pattern in the input separately. Note that this function will be 
        called at different intervals for synchronous and synchronous modes; 
        for asynchronous operation, it will be called after every unit is 
        :return: A matrix of the same dimensions as the input matrix
        containing the recalled patterns.
        """

        # check values in matrices match
        assert np.all(np.unique(Xd) == [0, 1]) \
            if sparse else np.all(np.unique(Xd) == [-1, 1])

        max_iter = 10 * np.log(self._w.shape[0]) \
            if max_iter is None else max_iter

        theta = sparse_theta if sparse else 0

        if mode == 'asynchronous':
            def recall_fn(pattern, cb):
                return self._asynchronous_recall(pattern, sparse, theta,
                                                 random_units, cb)
        elif mode == 'synchronous':
            def recall_fn(pattern, cb):
                return self._synchronous_recall(pattern, sparse, theta, cb)
        else:
            raise RuntimeError(f'Invalid mode: {mode}!')

        Y = np.empty(shape=Xd.shape)

        for i, pattern in enumerate(Xd):
            new_y = pattern.copy()
            convergence_count = 0
            iterations = 0
            while True:

                def epoch_cb(unit, pat):
                    return callback(iterations + 1, i, unit, pat)

                prev_y = new_y.copy()
                new_y = recall_fn(prev_y, epoch_cb)

                iterations += 1
                convergence_count = convergence_count + 1 \
                    if np.all(np.isclose(new_y, prev_y)) else 0

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

    def _energy(self, x: np.ndarray) -> float:
        """
        Calculates the energy for a given state or pattern.

        :param x: Input pattern or state of the network for which we want to
        calculate the energy.
        :return: Value of the energy.
        """
        I, J = self._w.shape

        energy = 0
        for i in range(I):
            for j in range(J):
                energy += - self._w[i, j] * x[i] * x[j]
        return energy


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

    print(nn._energy(x2d))

    print(Xp == X)
