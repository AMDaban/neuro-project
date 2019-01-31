class Shape:
    def __init__(self, rows, columns):
        self.rows = rows
        self.columns = columns


class GaborParameters:
    def __init__(self, shape, sigma, theta, lmd, gamma, psi):
        self.shape = shape
        self.sigma = sigma
        self.theta = theta
        self.lmd = lmd
        self.gamma = gamma
        self.psi = psi
