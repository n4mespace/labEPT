import numpy as np
from scipy.stats import t, f
import random
from typing import Tuple

np.set_printoptions(precision=3)


class FactorExperiment:
    '''
    Lab3: Three-factor experiment with linear regression

    '''
    def __init__(self,
                 X1_range: Tuple[float, float],
                 X2_range: Tuple[float, float], 
                 X3_range: Tuple[float, float], 
                 p: float) -> None:
        '''
        Init with variant constants
        
        Arguments:
            X1_range {Tuple[float, float]} -- [x_min_1, x_max_1]
            X2_range {Tuple[float, float]} -- [x_min_2, x_max_2]
            X3_range {Tuple[float, float]} -- [x_min_3, x_max_3]
            p {float} -- [probability]
        '''
        self.p = p
        self.N = 4
        self.M = 3

        self.x_mins = np.array(
            [X_range[0] for X_range
            in [X1_range, X2_range, X3_range]])
        self.x_maxs = np.array(
            [X_range[1] for X_range
            in [X1_range, X2_range, X3_range]])

        self.x_mean_min = np.mean(self.x_mins)
        self.x_mean_max = np.mean(self.x_maxs)

        self.y_min = 200 + self.x_mean_min
        self.y_max = 200 + self.x_mean_max

    def experiment(self) -> None:
        '''
        Run all needed operations

        '''
        self.create_plan_matrix()
        self.create_norm_matrix()

        self.f1 = self.M - 1
        self.f2 = self.N
        self.f3 = self.f1 * self.f2

        self.find_b()

        print('\nCohren criterion:')
        if not self.cohren_criterion():
            print(f'\nWrong!')
            print('Change m={self.M} to m\'={self.M+1}\n')
            self.M += 1
            self.experiment()

        self.student_criterion()
        
        self.f4 = self.N - self.d if self.N != self.d else 1
        self.fisher_criterion()

        # Regression
        print()
        print(f'y = {self.b[0]:.2f} + {self.b[1]:.2f}*x1 + ' +
              f'{self.b[2]:.2f}*x2 + {self.b[3]:.2f}*x3')


    def create_plan_matrix(self) -> None:
        '''
        Creates x and y arrays for plan experiment

        '''
        self.x = np.array(
            [[random.random() * (self.x_maxs[i] - self.x_mins[i]) + self.x_mins[i]
              for i in range(len(self.x_mins))]
             for j in range(self.N)]
        )
        self.y = np.array(
            [[random.random() * (self.y_max - self.y_min) + self.y_min
              for i in range(self.M)]
             for j in range(self.N)]
        )

        print('Plan matrix:')
        print('\nX values:')
        print(self.x)
        print('\nY values:')
        print(self.y)

    def create_norm_matrix(self) -> None:
        '''
        Normalize x array

        '''
        self.x_norm = np.ndarray((self.x.shape[0], self.x.shape[1] + 1))
        x0 = (self.x_maxs + self.x_mins) / 2
        dx = x0 - self.x_mins

        for i in range(self.x.shape[0]):
            for j in range(self.x.shape[1]):
                self.x_norm[i][1 + j] = (self.x[i][j] - x0[j]) / dx[j]
            self.x_norm[i][0] = 1

        print('\nNorm matrix:\n', self.x_norm)

    def find_b(self) -> None:
        '''
        Finds b-coefs for regression

        '''
        self.y_means = self.y.mean(axis=1)

        mx1 = np.mean(self.x.T[0])
        mx2 = np.mean(self.x.T[1])
        mx3 = np.mean(self.x.T[2])

        my = np.mean(self.y_means)

        a1 = np.mean(self.x.T[0] * self.y_means)
        a2 = np.mean(self.x.T[1] * self.y_means)
        a3 = np.mean(self.x.T[2] * self.y_means)

        a11 = np.mean(self.x.T[0] ** 2)
        a22 = np.mean(self.x.T[1] ** 2)
        a33 = np.mean(self.x.T[2] ** 2)

        a12 = a21 = np.mean(self.x.T[0] * self.x.T[1])
        a13 = a31 = np.mean(self.x.T[0] * self.x.T[2])
        a23 = a32 = np.mean(self.x.T[1] * self.x.T[2])

        self.b = np.linalg.solve([[1, mx1, mx2, mx3],
                                  [mx1, a11, a12, a13],
                                  [mx2, a21, a22, a23],
                                  [mx3, a31, a32, a33]],
                                 [my, a1, a2, a3])

        print(f'\nb coefs are:\n', self.b)
        print()

        regr = self.b[0] + self.x @ self.b[1:].T

        for i, (y_i, y_i_mean) in enumerate(zip(regr, self.y_means)):
            print(f'y{i+1} = {y_i:.3f}, y{i+1} mean = {y_i_mean:.3f}')

    def get_cohren_critical(self) -> float:
        '''
        Get table value of Cohren criterion
        
        Returns:
            float -- [criterion value]
        '''
        f_crit = f.isf((1 - self.p) / self.f2,
                       self.f1,
                       (self.f2 - 1) * self.f1)
        return f_crit / (f_crit + self.f2 - 1)

    def cohren_criterion(self) -> bool:
        '''
        Checks Cohren's criterion
        
        Returns:
            bool -- [criterion result]
        '''
        self.variances = np.var(self.y, axis=1) * self.M
        Gp = max(self.variances) / sum(self.variances)
        Gt = self.get_cohren_critical()

        print(f'Gp: {Gp:.3f} Gt: {Gt:.3f}')
        return Gp < Gt
        
    def student_criterion(self) -> None:
        '''
        Checks Student's criterion

        '''
        self.s2_b = np.mean(self.variances) / (self.M * self.N)
        
        s_b = np.sqrt(self.s2_b)
        b = np.abs(np.mean(self.x_norm * self.variances, axis=0))

        t_s = b / s_b
        t_tabl = round(t.ppf((1 + self.p) / 2, self.f3), 3)

        print('\nStudent criterion:')
        print('Values for factors:\n', t_s)
        print('nFt:\n', t_tabl)

        print('\nValuables:')
        valuable = t_s > t_tabl
        self.d = sum(valuable)

        for i in range(self.N):
            print(f'X{i} is valuable: {valuable[i]}')

        self.b *= valuable
        self.yh = self.b[0] + self.x @ self.b[1:].T

        print('\nValues for y with significant factors:\n', self.yh)

    def fisher_criterion(self) -> None:
        '''
        Checks Fisher's criterion

        '''
        nd_dif = self.N - self.d if self.N != self.d else 1
        s2_ad = sum([(self.yh[i] - self.y_means[i]) ** 2 for i in range(self.N)]) * self.M / nd_dif

        Fp = s2_ad / self.s2_b
        Ft = f.ppf(self.p, self.f4, self.f3)

        print('\nFisher criterion:')
        if Fp > Ft:
            print(f'OK with q = {1 - self.p:.2f}')
        else:
            print(f'Wrong with q = {1 - self.p:.2f}')



if __name__ == '__main__':
    # My variant X ranges
    X1_range = [-20, 15]
    X2_range = [-30, 45]
    X3_range = [-30, -15]

    # and probability
    p = 0.95

    # Init and run lab3 experiment    
    lab_3 = FactorExperiment(X1_range, X2_range, X3_range, p)
    lab_3.experiment()
