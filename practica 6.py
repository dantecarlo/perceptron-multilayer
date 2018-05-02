import math

class Perceptron(object):
    # Paramoetros de ingreso son la entidad, n y los pesos
    def __init__(self, ent, n, w):
        self.ent = ent
        self.n = n
        self.w = w
        self.y = [0] * len(self.ent)
        self.end = False
        self.cont = 0

    # Sumatoria de todos en la misma fila exepto el utlimo porque es el D
    def fun_trans(self):
        for x in range(0, len(self.ent)):
            y_aux = 0
            for y in range(0, len(self.ent[x]) - 1):
                y_aux = y_aux + self.ent[x][y] * w[y]
            self.y[x] = y_aux

    # Evalua si el y es positivo o negativo para dar un reusltado 0 o 1
    def fun_activ(self):
        for x in range(0, len(self.y)):
            if self.y[x] <= 0:
                self.y[x] = 0
            else:
                self.y[x] = 1

    # ajusta los pesos para cada iteracion si esta no es el valor esperado
    def fit(self, x):
        for y in range(0, len(self.w)):
            self.w[y] = self.w[y] + self.n * (self.ent[x]
                                     [len(self.ent[x]) - 1] - self.y[x]) * self.ent[x][y]

    def run(self):
        self.fun_trans()
        self.fun_activ()
        finish = [True] * len(self.ent)
        while self.end is False:
            self.end = True
            for x in range(0, len(self.y)):
                if self.ent[x][len(self.ent[x]) - 1] != self.y[x]:
                    self.fit(x)
                    finish[x] = False
                    self.fun_trans()
                    self.fun_activ()
                    print(self.w)
                    # self.plo(w)
                else:
                    finish[x] = True
                    print(self.w)
            for x in range(0, len(finish)):
                self.end = self.end and finish[x]
            print(self.end)


entity = [[1, 0, 0, 0], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 1]]
w = [0, 0, 0]
perc = Perceptron(entity, 0.5, w)
perc.run()


class multilayer(object):
    def __init__(self, w, x, d, n, p_h, p_f):
        self.w = w
        self.x = x
        self.d = d
        self.n = n
        self.p_h = p_h
        self.p_f = p_f
        self.u = []

    def fx(self, x):
        res = 1 / (1 + math.exp(-x))
        return res

    def der_fx(self, x):
        res = math.exp(x) / (1 + math.exp(x))**2
        return res

    # set de sum of w*x
    def set_u(self):
        for i in range(0, len(self.w)):
            sum = 0
            for j in range(0, len(self.w)):
                sum = sum + (self.w[j] * self.x[j])
            self.u.appent(sum)

    # fit de weigth for the hidden layers
    def fit_h(self):
        for i in range(0, len(self.w)):
            e = self.der_fx(self.u[i])
            self.w[i] = self.w[i] + self.n * self.x[i] * e
