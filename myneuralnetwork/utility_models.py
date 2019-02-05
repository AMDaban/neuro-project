class Shape:
    def __init__(self, rows, columns):
        self.rows = rows
        self.columns = columns

    def get_tuple(self):
        return self.rows, self.columns


class GaborParameters:
    def __init__(self, shape, sigma, theta, lmd, gamma, psi):
        self.shape = shape
        self.sigma = sigma
        self.theta = theta
        self.lmd = lmd
        self.gamma = gamma
        self.psi = psi


# IF stands for integrate and fire
class IF:
    def __init__(self, threshold):
        self._threshold = threshold
        self._potential = 0
        self._hit_threshold = False

    def change_potential(self, diff):
        self._potential += diff

        if self._potential > self._threshold:
            self._potential = 0
            self._hit_threshold = True

    def hit_threshold(self):
        return self._hit_threshold

    def reset(self):
        self._potential = 0
        self._hit_threshold = False
