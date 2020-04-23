import numpy as np
import pandas as pd
from scipy.stats import t, f
import random
import sklearn.linear_model as lm
from typing import Tuple, TypeVar, Callable
from functools import wraps
from time import time


np.set_printoptions(precision=3)
pd.set_option('display.precision', 3)
RT = TypeVar('RT')  # generic return type 


class FactorExperimentDecorators:
    '''
    Helper decorators for FactorExperiment

    '''
    @classmethod
    def timeit(cls, func: Callable[..., RT]) -> Callable[..., RT]:
        '''
        Measure time for statistical checks
        
        Decorators:
            wraps - [for debugging things]
        
        Arguments:
            func {Callable[..., RT]} -- [function to measure]
        
        Returns:
            {Callable[..., RT]} -- [wrappped function]
        '''

        @wraps(func)
        def _wrapper(*args, **kwargs) -> RT:
            '''
            Return result of function and prints time of execution
            
            Arguments:
                *args {[type]} -- [*args of function]
                **kwargs {[type]} -- [*kwargs of function]
            
            Returns:
                [type] -- [result of function]
            '''
            start = time()
            try:
                return func(*args, **kwargs)
            finally:
                end = (time() - start) * 1000
                print(f'Total execution time: {end:.3f} ms')

        return _wrapper


class FactorExperiment:
    '''
    Lab5: Three-factor experiment with quadratic and interaction effects
    
    Variables:
        l: float {number} -- [the magnitude of the star shoulder]
        x_plan {[type]} -- [plan matrix for k = 3]
    '''
    l: float = 1.215
    x_plan = np.array(
          [[-1, -1, -1],
           [-1, -1, 1],
           [-1,  1, -1],
           [-1,  1, 1],
           [1, -1, -1],
           [1, -1,  1],
           [1,  1, -1],
           [1,  1, 1],
           [-l, 0, 0],
           [l, 0, 0],
           [0, -l, 0],
           [0, l, 0],
           [0, 0, -l],
           [0, 0, l],
           [0, 0, 0]]
        )

    def __init__(self,
                 X1_range: Tuple[int, int],
                 X2_range: Tuple[int, int], 
                 X3_range: Tuple[int, int], 
                 p: float = 0.95) -> None:
        '''
        Init with variant constants
        
        Arguments:
            X1_range {Tuple[float, float]} -- [x_min_1, x_max_1]
            X2_range {Tuple[float, float]} -- [x_min_2, x_max_2]
            X3_range {Tuple[float, float]} -- [x_min_3, x_max_3]
            p {float} -- [probability for statistical tests]
        '''
        self.p = p
        self.N = 15
        self.M = 3

        X_ranges = np.array([X1_range, X2_range, X3_range])

        self.x_mins = X_ranges[:, 0]
        self.x_maxs = X_ranges[:, 1]

        self.x_mean_min = np.mean(self.x_mins)
        self.x_mean_max = np.mean(self.x_maxs)

        self.y_min = 200 + self.x_mean_min
        self.y_max = 200 + self.x_mean_max

    def experiment(self) -> None:
        '''
        Run all needed operations

        '''
        self.create_naturalized_matrix()

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
        print('Stable variances!')

        self.student_criterion()
        
        self.f4 = self.N - self.d if self.N != self.d else 1
        self.fisher_criterion()

        self.print_regression()

    def print_regression(self) -> None:
        '''
        Prints regression with significance coefs

        '''
        print(f'\ny = {self.b[0]:.2f}', end='')

        for i in range(1, len(self.b)):
            if self.b[i]:
                print(f' + {self.b[i]:.2f}*{self.x.columns[i]}', end='')

    def create_naturalized_matrix(self) -> None:
        '''
        Creates x and y arrays for plan experiment

        '''
        x0 = (self.x_maxs + self.x_mins) / 2
        dx = x0 - self.x_mins

        self.x = np.array(
            [[self.x_plan[j, i] * dx[i] - x0[i] 
              if j > 8 
              else (self.x_mins[i] 
                 if self.x_plan[j, i] == -1 
                 else self.x_maxs[i])
              for i in range(len(self.x_mins))]
             for j in range(self.N)]
        )

        self.x = pd.DataFrame(self.x, columns=['x1', 'x2', 'x3'])
        self.x['b'] = 1
        self.x = self.x.reindex(columns=['b', 'x1', 'x2', 'x3'])

        self.x['x1_x2'] = self.x.x1 * self.x.x2
        self.x['x1_x3'] = self.x.x1 * self.x.x3
        self.x['x2_x3'] = self.x.x2 * self.x.x3
        self.x['x1_x2_x3'] = self.x.x1 * self.x.x2 * self.x.x3
        
        self.x['x1_2'] = self.x.x1 ** 2
        self.x['x2_2'] = self.x.x2 ** 2
        self.x['x3_2'] = self.x.x3 ** 2

        self.y = np.array(
            [[random.random() * (self.y_max - self.y_min) + self.y_min
              for i in range(self.M)]
             for j in range(self.N)]
        )

        print('Plan matrix:')
        print('X values:')
        print(self.x)
        print('\nY values:')
        print(self.y)

    def find_b(self) -> None:
        '''
        Finds b-coefs for regression

        '''
        self.y_means = self.y.mean(axis=1)

        regression = lm.LinearRegression(fit_intercept=False)
        regression.fit(self.x, self.y_means)

        self.b = regression.coef_
        print(f'\nb coefs are:\n', self.b)
        print()

        self.check_regression()

    def check_regression(self) -> None:
        '''
        Compare regression results and true values
        
        Arguments:
            regression_values {[type]} -- [regression values]
        '''
        self.yh = self.b[0] + self.x.drop('b', axis=1) @ self.b[1:].T
        
        for i, (y_i, y_i_mean) in enumerate(zip(self.yh, self.y_means)):
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

    @FactorExperimentDecorators.timeit
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
        
    @FactorExperimentDecorators.timeit
    def student_criterion(self) -> None:
        '''
        Checks Student's criterion

        '''
        self.s2_b = np.mean(self.variances) / (self.M * self.N)
        
        s_b = np.sqrt(self.s2_b)
        b = np.abs(np.mean((self.variances * self.x.T).T, axis=0))

        t_s = b / s_b
        t_tabl = round(t.ppf((1 + self.p) / 2, self.f3), 3)

        print('\nStudent criterion:')
        print('Values for factors:\n', t_s)
        print('\nFt:\n', t_tabl)

        print('\nValuables:')
        valuable = t_s > t_tabl
        self.d = sum(valuable)

        for i in range(self.x.shape[1]):
            print(f'Coef_{i} is valuable: {valuable[i]}')

        self.b *= valuable

        print('\nValues for y with significant factors:\n')
        self.check_regression()

    @FactorExperimentDecorators.timeit
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
    X1_range = (-10, 9)
    X2_range = (0, 1)
    X3_range = (-3, 4)

    # Init and run lab5 experiment    
    lab_5 = FactorExperiment(X1_range, X2_range, X3_range)
    lab_5.experiment()