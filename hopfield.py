import warnings
from typing import Optional

import numpy as np
from numpy.random import default_rng

rand_gen = default_rng()


class HopfieldNetwork:
    def __init__(self):
        super(HopfieldNetwork, self).__init__()
        self._w = np.empty(0)

    def train(self, X: np.ndarray, convergence_threshold: int = 5) -> None:
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

    def recall(self, X: np.ndarray,
               synchronous: bool = True,
               convergence_threshold: int = 5,
               max_iter: Optional[int] = None) -> np.ndarray:

        Y = np.empty(shape=X.shape)

        max_iter = 10 * np.log(self._w.shape[0]) \
            if max_iter is None else max_iter

        if synchronous:
            # converge per pattern, not the whole matrix
            for i, pattern in enumerate(X):
                new_y = pattern.copy()
                convergence_count = 0
                iterations = 0

                while True:
                    prev_y = new_y.copy()
                    new_y = np.dot(self._w, prev_y)
                    new_y[new_y >= 0] = 1  # x >= 0 -> 1
                    new_y[new_y < 0] = -1  # x < 0 -> -1

                    convergence_count = convergence_count + 1 \
                        if np.all(np.isclose(new_y, prev_y)) else 0

                    iterations += 1

                    if convergence_count >= convergence_threshold:
                        break
                    elif iterations >= max_iter:
                        warnings.warn(f'Pattern {i} did not converge after '
                                      f'the maximum number of iterations ('
                                      f'{int(np.floor(max_iter)):d})!',
                                      Warning)
                        break

                Y[i, :] = new_y

            return Y
        else:
            raise RuntimeError('Not implemented yet.')


if __name__ == '__main__':
    x1 = [-1, -1, 1, -1, 1, -1, -1, 1]
    x2 = [-1, -1, -1, -1, -1, 1, -1, -1]
    x3 = [-1, 1, 1, -1, -1, 1, -1, 1]
    X = np.array([x1, x2, x3])

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
