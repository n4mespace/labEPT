import random
from prettytable import PrettyTable
from numpy import mean
from typing import List


class FactorExperiment:
    def __init__(self, n: int, a: List[float]) -> None:
        self.n = n  # Num of dotes
        self.a = a  # List of coefs

        # Factors
        self.X_1 = self.random_list()
        self.X_2 = self.random_list()
        self.X_3 = self.random_list()
        self.factors = [self.X_1, self.X_2, self.X_3]

        # Normalized Factors
        self.Xn_1 = self.get_xn(self.X_1)
        self.Xn_2 = self.get_xn(self.X_2)
        self.Xn_3 = self.get_xn(self.X_3)

        # Form regression and sort Y[i]
        self.Y = self.form_regression()
        self.sorted_Y = sorted(self.Y)

        # Form zero Factors
        self.x0 = self.form_x0()

        # Form Factors change interval
        self.dx = self.form_dx()

        # Form Y etalon
        self.Y_et = self.form_Y_et()

        # Form answer by variant |<-mean(Y)|
        self.answear = self.form_answear()

        # Form Plans point
        self.point = self.form_point()

    def random_list(self) -> List[float]:
        ''' Gen Factor vector[n] '''
        return [random.randint(0, 20) for _ in range(self.n)]

    def get_xn(self, X: List[float]) -> List[float]:
        ''' Normalize Factor vector X '''
        a = sorted(X)
        x0 = (a[-1] + a[0]) / 2
        dx = x0 - a[0]
        return [round((X[i] - x0) / dx, 2) for i in range(self.n)]

    def form_regression(self) -> List[float]:
        ''' Get regression with given Factors '''
        return [self.a[0] + self.a[1] * self.X_1[i] +
                self.a[2] * self.X_2[i] + self.a[3] * self.X_3[i]
                for i in range(self.n)]

    def form_x0(self) -> List[float]:
        ''' Calculate x0 by given Factors '''
        return [round((X[0] + X[-1]) / 2, 2)
                for X in self.factors]

    def form_dx(self) -> List[float]:
        ''' Calculate dx by given Factors '''
        return [(self.x0[i] - sorted(self.factors[i])[0])
                for i in range(len(self.factors))]

    def form_Y_et(self) -> float:
        ''' Criteria by variant '''
        return mean(self.a[0] + self.a[1] * self.x0[0] +
                    self.a[2] * self.x0[1] + self.a[3] * self.x0[2])

    def form_answear(self) -> float:
        ''' Get answear by criteria Y_et '''
        return [self.sorted_Y[i] for i in range(self.n)
                if self.sorted_Y[i] > self.Y_et][0]

    def form_point(self) -> List[float]:
        ''' Get point which satisfied criteria answear '''
        return [(self.X_1[self.Y.index(self.sorted_Y[i])],
                 self.X_2[self.Y.index(self.sorted_Y[i])],
                 self.X_3[self.Y.index(self.sorted_Y[i])])
                for i in range(self.n) if self.sorted_Y[i] > self.Y_et][0]

    def check_results(self) -> None:
        ''' Print results to console '''
        experiment = PrettyTable()
        experiment.field_names = ['№', 'X1', 'X2', 'X3', 'Y',
                                  'Хn1', 'Хn2', 'Хn3']
        for i in range(self.n):
            experiment.add_row([i+1, self.X_1[i], self.X_2[i], self.X_3[i],
                                self.Y[i], self.Xn_1[i], self.Xn_2[i],
                                self.Xn_3[i]])

        characteristic = PrettyTable()
        characteristic.field_names = ['Characteristic', 'X1', 'X2', 'X3']
        characteristic.add_row(['x0', self.x0[0], self.x0[1], self.x0[2]])
        characteristic.add_row(['dx', self.dx[0], self.dx[1], self.dx[2]])

        result = PrettyTable()
        result.field_names = ['Result', 'Value']
        result.add_row(['Y', f'{self.a[0]} + {self.a[1]}*x1 + {self.a[2]}*x2 + {self.a[3]}*x3'])
        result.add_row(['Y_etalon', f'{self.answear}'])
        result.add_row(['Point', f'{self.point}'])
        print(experiment)
        print(characteristic)
        print(result)


if __name__ == '__main__':
    n = 8
    a = [1, 4, 3, 2]
    lab_1 = FactorExperiment(n, a)
    lab_1.check_results()
