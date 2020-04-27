import numpy as np
import pandas as pd
from scipy.stats import t, f
import sklearn.linear_model as lm
from typing import Tuple, TypeVar, Callable, Any, Union
from functools import wraps
from time import time


np.set_printoptions(precision=3)
pd.set_option('display.precision', 3)

RT = TypeVar('RT')                # generic return type 
X = Union[np.array, pd.Series]    # generic args type  for variant function
Y = Union[np.array, pd.Series]


class FactorExperimentDecorators:
    '''
    Helper decorators for FactorExperiment

    '''
    @classmethod
    def timeit(cls, func: Callable[..., RT]) -> Callable[..., RT]:
        '''
        Measure time for statistical checks
        
        Decorators:
            wraps - for debugging things
        
        Arguments:
            func {Callable[..., RT]} -- function to measure
        
        Returns:
            {Callable[..., RT]} -- wrappped function
        '''

        @wraps(func)
        def _wrapper(*args: Any, **kwargs: Any) -> RT:
            '''
            Return result of function and prints time of execution
            
            Arguments:
                *args {Any} -- args of function
                **kwargs {Any} -- kwargs of function
            
            Returns:
                RT -- result of function
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
    Lab6: Three-factor experiment with quadratic and interaction effects
    
    Variables:
        l: float {number} -- the magnitude of the star shoulder
        x_plan {np.array} -- plan matrix for k = 3
    '''
    l: float = 1.73
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
                 func: Callable[[X, X, X], Y],
                 p: float = 0.95) -> None:
        '''
        Init with variant constants
        
        Arguments:
            X1_range {Tuple[float, float]} -- [x_min_1, x_max_1]
            X2_range {Tuple[float, float]} -- [x_min_2, x_max_2]
            X3_range {Tuple[float, float]} -- [x_min_3, x_max_3]
            func {Callable[[X, X, X], Y]} -- variant function for y generation
            p {float} -- probability for statistical tests
        '''
        self.p = p
        self.N = 15
        self.M = 3

        X_ranges = np.array([X1_range, X2_range, X3_range])

        self.x_mins = X_ranges[:, 0]
        self.x_maxs = X_ranges[:, 1]

        self.func = func

    def experiment(self) -> None:
        '''
        Run all needed operations

        '''
        self.create_naturalized_matrix()

        self.f1 = self.M - 1
        self.f2 = self.N
        self.f3 = self.f1 * self.f2

        self.find_b()

        if not self.cohren_criterion():
            print(f'\nWrong!')
            print('Change m={self.M} to m\'={self.M+1}\n')
            self.M += 1
            self.experiment()

        self.student_criterion()
        
        self.f4 = self.N - self.d if self.N != self.d else 1
        self.fisher_criterion()

        self.print_regression()
        print('\nEnd of factor experiment')

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

        self.generate_y()

        print('Plan matrix:')
        print('X values:\n', self.x)
        print('\nY values:\n', self.y)

    def generate_y(self) -> None:
        '''
        Using variant function generate y

        '''
        self.y = self.func(self.x.x1, self.x.x2, self.x.x3)
        self.y += np.random.uniform(0, 10, size=self.y.shape) - 5

    def find_b(self) -> None:
        '''
        Finds b-coefs for regression

        '''
        regression = lm.LinearRegression(fit_intercept=False)
        regression.fit(self.x, self.y)

        self.b = regression.coef_
        print(f'\nb coefs are:\n', self.b, '\n')

        self.check_regression()

    def check_regression(self) -> None:
        '''
        Compare regression results and true values

        '''
        self.yh = self.b[0] + self.x.drop('b', axis=1) @ self.b[1:].T
        self.results = pd.DataFrame({
            'y_pred': self.yh,
            'y_true': self.y
        })
        print('Results:\n', self.results)

    def get_cohren_critical(self) -> float:
        '''
        Get table value of Cohren criterion
        
        Returns:
            float -- criterion value
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
            bool -- criterion result
        '''
        print('\nCohren criterion:')

        self.variances = np.var(self.y) * self.M
        Gp = max(self.y) / sum(self.y)
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
        t_tabl = t.ppf((1 - self.p) / 2, self.f3)

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
        
        if not all(valuable):
            self.check_regression()

    @FactorExperimentDecorators.timeit
    def fisher_criterion(self) -> None:
        '''
        Checks Fisher's criterion

        '''
        nd_dif = self.N - self.d if self.N != self.d else self.N - self.d + 1
        s2_ad = ((self.yh - self.y) ** 2).sum() * self.M / nd_dif

        Fp = s2_ad / self.s2_b
        Ft = f.isf(1 - self.p, self.f4, self.f3)

        print('\nFisher criterion:')
        print(f'Fp: {Fp},  Ft: {Fp}')

        if Fp > Ft or (Fp - Ft) < 0.00000000001:
            print(f'OK with q = {1 - self.p:.2f}')
        else:
            print(f'Wrong with q = {1 - self.p:.2f}')



if __name__ == '__main__':
    # My variant X ranges
    X1_range = (-20, 15)
    X2_range = (-30, 45)
    X3_range = (-30, -15)

    # and function
    func = lambda x1, x2, x3: (5.4 + 3.4 * x1 + 9.6 * x2 + 6.8 * x3 
                               + 0.8 * x1 * x2 + 0.8 * x1 * x3 + 9.9 * x2 * x3 + 4.5 * x1 * x2 * x3
                               + 3.1 * x1**2 + 0.1 * x2**2 + 1.2 * x3**2)


    # Init and run lab6 experiment    
    lab_6 = FactorExperiment(X1_range, X2_range, X3_range, func)
    lab_6.experiment()
