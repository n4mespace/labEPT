import numpy as np
from typing import List

np.set_printoptions(precision=3)


class FactorExperiment:
    def __init__(self,
                 X1_range: List[int],
                 X2_range: List[int],
                 Y_range: List[int], m: int) -> None:
        # Criteria for different m with confidence p=0.99
        self.R_critical = {
            5: 2,
            6: 2.16,
            7: 2.3,
            8: 2.43,
            9: 2.5
        }

        # Min and max for X1, X2, Y2
        self.X1_range = X1_range
        self.X2_range = X2_range
        self.Y_range = Y_range

        # Generate random plan matrix in X's ranges
        self.plan_matrix = np.array(
            [np.random.randint(*self.X1_range, size=3),
             np.random.randint(*self.X2_range, size=3)]).T

        self.x0 = [np.mean(self.X1_range), np.mean(self.X2_range)]

        # Normalize plan matrix
        self.norm_matrix = self.make_norm_plan_matrix()

        # Number of y experiments for the same factors comb
        self.m = m

        # Running experiment with given m
        self.experiment()

        # Find coefs
        self.b = self.find_b()
        self.a = self.find_a()

        # Checking coefs
        # Must be close to Y means
        self.check_b = self.check_b_koefs()
        self.check_a = self.check_a_koefs()

    def experiment(self):
        # Generate y matrix
        self.y_matrix = np.random.randint(*self.Y_range, size=(3, self.m))
        self.y_mean = np.mean(self.y_matrix, axis=1)

        self.y_var = np.var(self.y_matrix, axis=1)
        self.sigma = np.sqrt((2 * (2 * self.m - 2)) / (self.m * (self.m - 4)))

        # Checking variance with Romanovskiy's theorem
        if not self.check_r():
            print(f'\nCriteria wrong! Change m={self.m} to m\'={self.m+1}\n')
            self.m += 1
            self.experiment()

    def make_norm_plan_matrix(self) -> np.array:
        self.N = self.plan_matrix.shape[0]
        self.k = self.plan_matrix.shape[1]

        interval_of_change = [self.X1_range[1] - self.x0[0],
                              self.X2_range[1] - self.x0[1]]
        X_norm = [
            [(self.plan_matrix[i, j] - self.x0[j]) / interval_of_change[j]
             for j in range(self.k)]
            for i in range(self.N)
        ]
        return np.array(X_norm)

    def check_r(self) -> bool:
        for i in range(len(self.y_var)):
            for j in range(len(self.y_var)):
                if i > j:
                    if self.y_var[i] >= self.y_var[j]:
                        R = (abs((self.m - 2) * self.y_var[i] /
                             (self.m * self.y_var[j]) - 1) / self.sigma)
                    else:
                        R = (abs((self.m - 2) * self.y_var[j] /
                             (self.m * self.y_var[i]) - 1) / self.sigma)
                    if R > self.R_critical[self.m]:
                        print('Variance isn\'t stable!')
                        return False
        return True

    def find_b(self) -> np.array:
        mx1 = np.mean(self.norm_matrix[:, 0])
        mx2 = np.mean(self.norm_matrix[:, 1])

        a1 = np.mean(self.norm_matrix[:, 0] ** 2)
        a2 = np.mean(self.norm_matrix[:, 0] * self.norm_matrix[:, 1])
        a3 = np.mean(self.norm_matrix[:, 1] ** 2)

        my = np.mean(self.y_mean)
        a11 = np.mean(self.norm_matrix[:, 0] * self.y_mean)
        a22 = np.mean(self.norm_matrix[:, 1] * self.y_mean)

        b = np.linalg.solve([[1, mx1, mx2],
                             [mx1, a1, a2],
                             [mx2, a2, a3]],
                            [my, a11, a22])
        return b

    def find_a(self) -> np.array:
        # Naturalization
        delta_x = [abs(self.X1_range[1] - self.X1_range[0]) / 2,
                   abs(self.X2_range[1] - self.X2_range[0]) / 2]
        a = [(self.b[0] - self.b[1] * self.x0[0] / delta_x[0] -
              self.b[2] * self.x0[1] / delta_x[1]),
             self.b[1] / delta_x[0],
             self.b[2] / delta_x[1]]
        return np.array(a)

    def check_b_koefs(self) -> np.array:
        return np.array([
            (self.b[0] + np.sum(self.b[1:3] * self.norm_matrix[i]))
            for i in range(self.N)])

    def check_a_koefs(self) -> np.array:
        return np.array([
            (self.a[0] + np.sum(self.a[1:3] * self.plan_matrix[i]))
            for i in range(self.N)])

    def check_results(self) -> None:
        print('Plan matrix:\n', self.plan_matrix)
        print('Norm matrix:\n', self.norm_matrix)
        print('Y matrix:\n', self.y_matrix)
        print('\nNormalized koefs: ', self.b)
        print('Naturalized koefs:', self.a)
        print('\nY mean:           ', self.y_mean)
        print('Check b:          ', self.check_b)
        print('Check a:          ', self.check_a)


if __name__ == '__main__':
    m = 5
    X1_range = [-20, 15]
    X2_range = [-30, 45]
    Y_range = [110, 210]
    lab_2 = FactorExperiment(X1_range, X2_range, Y_range, m)
    lab_2.check_results()
