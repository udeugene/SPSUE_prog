import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from matplotlib import style
from scipy.misc import derivative
import inspect

from IPython.display import display, clear_output

style.use('ggplot')
style.use('seaborn-ticks')

def fib(n, cache = {0: 0, 1: 1}):
    if n not in cache:
        cache[n] = fib(n-1, cache) + fib(n-2, cache)
    return cache[n]

def D(f, x):
    return derivative(f, x, dx=1e-6)


def D2(f, x):
    return derivative(f, x, n=2, dx=1e-6)


class Passive_search:
    def __init__(self, f, a, b, e, prec=4):
        raw_f = inspect.getsource(f)
        self.func = raw_f[raw_f.find('return') + 6:].strip()
        self.bounds = [a, b]
        self.e = e
        self.prec = prec
        self.f = f

    def data(self):
        f, a, b, e = self.f, self.bounds[0], self.bounds[1], self.e
        k = (b - a) / e
        i = 0
        points = {}
        while i <= k:
            x = a + ((b - a) / k) * i
            points[x] = f(x)
            i += 1
        data=points
        dfdata = pd.DataFrame(points, index = [0]).T
        dfdata = dfdata.reset_index()
        dfdata.columns = ['x', 'f(x)']
        self.dfdata = dfdata
        return data
    def res(self):
        points = self.data()
        return [min(points, key=points.get), points[min(points, key=points.get)]]

    def smooth_plot(self):
        f, a, b, e = self.f, self.bounds[0], self.bounds[1], self.e
        res = self.res()

        fig, ax = plt.subplots(figsize=(12, 6))
        x_space = np.linspace(a, b, round((b - a) * 100))
        ax.scatter(res[0], res[1], label=f'min f(x) = {round(res[1], self.prec)}')
        ax.plot(x_space, [f(x) for x in x_space], c='black', label=f'f(x) = {self.func}')

        ax.set_xlabel('X', size=15, fontweight=500)
        ax.set_ylabel('Y', size=15, fontweight=500)
        ax.legend(prop={'size': 14})
        return (ax)

    def rough_plot(self):
        f, a, b, e = self.f, self.bounds[0], self.bounds[1], self.e
        res = self.res()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(res[0], res[1], label=f'min f(x) = {round(res[1], self.prec)}')
        ax.plot(list(self.data().keys()), self.data().values(), c='black', label=f'f(x) = {self.func}')

        ax.set_xlabel('X', size=15, fontweight=500)
        ax.set_ylabel('Y', size=15, fontweight=500)
        ax.legend(prop={'size': 14})
        return (ax)

    def rep(self):
        return print(
            f'Минимум функции f(x) = {self.func}, равный {round(self.res()[1], self.prec)}  достигается при x = {round(self.res()[0], self.prec)} с точностью e = {self.e}')


class Golden_search:
    def __init__(self, f, a, b, e, prec=4):
        raw_f = inspect.getsource(f)
        self.func = raw_f[raw_f.find('return') + 6:].strip()
        self.bounds = [a, b]
        self.e = e
        self.prec = prec
        self.f = f

    def fit(self):
        f, a, b, e = self.f, self.bounds[0], self.bounds[1], self.e
        prec = self.prec

        T = (3 - math.sqrt(5)) / 2
        i = 0
        log_dict = {}
        x1 = a + T * (b - a)
        x2 = a + b - x1
        while 1:
            y1 = round(f(x1), prec)
            y2 = round(f(x2), prec)
            log_dict[i] = [a, b, x1, y1, x2, y2]
            i = i + 1
            if y1 <= y2:
                b = x2
                x2 = x1
                x1 = a + b - x2
            else:
                a = x1
                x1 = x2
                x2 = a + b - x1
            l = b - a
            if l <= e:
                x_opt = 0.5 * (a + b)
                log = pd.DataFrame(log_dict).T
                opt = [x_opt, f(x_opt)]
                log.columns = ['a', 'b', 'x1', 'y1', 'x2', 'y2']
                self.res = opt
                self.data = log
                break

    def rep(self):
        return print(
            f'Минимум функции f(x) = {self.func}, равный {round(self.res[1], self.prec)}  достигается при x = {round(self.res[0], self.prec)} с точностью e = {self.e} на шаге {len(self.data)}')

    def vis(self):
        df, opt = self.data, self.res
        f, a, b = self.f, self.bounds[0], self.bounds[1]
        prec = self.prec
        plt_x = np.linspace(a, b, round((b - a) * 250))
        plt_data = np.array([plt_x, [f(x) for x in plt_x]])
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.plot(plt_data[0], plt_data[1], label=f'f(x) = {self.func}')
        ax.scatter(opt[0], opt[1], c='red', label=f'f(x*) = {round(opt[1], prec)}')
        ax.legend(prop={'size': 14})
        ax.grid()
        return (ax)

    def vis_steps(self):
        df, opt = self.data, self.res
        f, a, b = self.f, self.bounds[0], self.bounds[1]
        prec = self.prec

        plt_x = np.linspace(a, b, round((b - a) * 250))
        plt_data = np.array([plt_x, [f(x) for x in plt_x]])
        fig, ax = plt.subplots(figsize=(16, 9))

        for i in range(5):
            display(fig)
            ax.cla()
            clear_output(wait=True)
            plt.pause(0.5)

        for i in range(len(df)):
            if i % 7 == 0 and i != 0:
                a, b = x1 - a / 4 * x1, x2 + b / 4
            ax.set_xlim(a, b)

            x1 = df['x1'][i]
            y1 = df['y1'][i]

            x2 = df['x2'][i]
            y2 = df['y2'][i]

            ax.scatter(x1, y1, c='white', label=f'step = {i + 1}')

            ax.axvline(df['a'][i], c='blue', label=f'a = {round(df["a"][i], prec)}')
            ax.axvline(df['b'][i], c='orange', label=f'b = {round(df["b"][i], prec)}')

            ax.scatter(x1, y1, marker='x', c='black', label=f'f(x1) = {y1}')

            ax.plot(plt_data[0], plt_data[1])
            ax.scatter(x2, y2, marker='x', c='red', label=f'f(x2) = {y2}')

            if i == len(df) - 1:
                ax.scatter(opt[0], opt[1], c='red', marker='|', label=f'f(x*) = {round(opt[1], prec)}')

            ax.legend(prop={'size': 14})
            ax.grid()
            display(fig)

            ax.cla()

            clear_output(wait=True)
            plt.pause(0.5)


class Fibonacci_search:
    def __init__(self, f, a, b, e, prec = 4):
        self.prec = prec
        raw_f = inspect.getsource(f)
        self.func = raw_f[raw_f.find('return') + 6:].strip()
        self.a, self.b, self.e = a, b, e
        self.f = f
        n = 1
        while (b - a) / e > fib(n):
            n += 1
        self.n = n

    def fit(self):
        a, b, e, n, f = self.a, self.b, self.e, self.n, self.f
        prec = self.prec
        la = a + (fib(n - 2) / fib(n)) * (b - a)
        mu = a + (fib(n - 1) / fib(n)) * (b - a)
        k = 1
        data = {0: [a, b, la, f(la), mu, f(mu)]}

        while k != n - 2:
            if f(la) > f(mu):
                a = la
                la = mu
                mu = a + (fib(n - k - 1) / fib(n - k)) * (b - a)
            else:
                b = mu
                mu = la
                la = a + (fib(n - k - 2) / fib(n - k)) * (b - a)
            data[k] = [a, b, la, f(la), mu, f(mu)]
            k += 1

        mu = la + e
        if f(la) < f(mu):
            b = mu
        else:
            a = la

        x_min = round((a + b) / 2, prec)
        res = [x_min, round(f(x_min), prec)]

        cache = pd.DataFrame(data).T
        cache.columns = ['a', 'b', 'la', 'f(la)', 'mu', 'f(mu)']

        self.data = cache
        self.res = res

    def rep(self):
        return print(
            f'Минимум функции f(x) = {self.func}, равный {round(self.res[1], self.prec)}  достигается при x = {round(self.res[0], self.prec)} с точностью e = {self.e} на шаге {len(self.data)}')

    def vis(self):
        df, opt = self.data, self.res
        f, a, b = self.f, self.a, self.b
        prec = self.prec
        plt_x = np.linspace(a, b, round((b - a) * 250))
        plt_data = np.array([plt_x, [f(x) for x in plt_x]])
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.plot(plt_data[0], plt_data[1], label=f'f(x) = {self.func}')
        ax.scatter(opt[0], opt[1], c='red', label=f'f(x*) = {round(opt[1], prec)}')
        ax.legend(prop={'size': 14})
        ax.grid()
        return (ax)

    def vis_steps(self):
        df, opt = self.data, self.res
        f, a, b = self.f, self.a, self.b
        prec = self.prec

        plt_x = np.linspace(a, b, round((b - a) * 250))
        plt_data = np.array([plt_x, [f(x) for x in plt_x]])
        fig, ax = plt.subplots(figsize=(16, 9))

        for i in range(5):
            display(fig)
            ax.cla()
            clear_output(wait=True)
            plt.pause(0.5)

        for i in range(len(df)):
            if i % 7 == 0 and i != 0:
                a, b = x1 - a / 8 * x1, x2 + b / 8
            ax.set_xlim(a - a / 5, b + b / 5)

            x1 = df['la'][i]
            y1 = df['f(la)'][i]

            x2 = df['mu'][i]
            y2 = df['f(mu)'][i]

            ax.scatter(x1, y1, c='white', label=f'step = {i + 1}')

            ax.axvline(df['a'][i], c='blue', label=f'a = {round(df["a"][i], prec)}')
            ax.axvline(df['b'][i], c='orange', label=f'b = {round(df["b"][i], prec)}')

            ax.scatter(x1, y1, marker='x', c='black', label=f'f(la) = {round(y1, prec)}')

            ax.plot(plt_data[0], plt_data[1])
            ax.scatter(x2, y2, marker='x', c='red', label=f'f(mu) = {round(y2, prec)}')

            if i == len(df) - 1:
                ax.scatter(opt[0], opt[1], c='red', marker='|', label=f'f(x*) = {round(opt[1], prec)}')

            ax.legend(prop={'size': 14})
            ax.grid()
            display(fig)

            ax.cla()

            clear_output(wait=True)
            plt.pause(0.5)


class Secant_search:
    def __init__(self, f, a, b, e, prec=4):
        raw_f = inspect.getsource(f)
        self.func = raw_f[raw_f.find('return') + 6:].strip()
        self.a, self.b, self.e = a, b, e
        self.prec = prec
        self.f = f

    def D(self, f, x):
        return derivative(f, x, dx=1e-6)

    def D2(self, f, x):
        return derivative(f, x, n=2, dx=1e-6)


    def fit(self):
        a, b, e, f = self.a, self.b, self.e, self.f
        prec = self.prec

        cache = {}
        i = 0
        if D(self.f, a) >= 0 or D(self.f, b) <= 0:
            return 'Невозможно применить метод'
        while 1:
            z = b - ((D(self.f, b) * (b - a)) / (D(self.f, b) - D(self.f, a)))
            cache[i] = [a, self.f(a), b, self.f(b), z, D(self.f, z), self.f(z)]
            if abs(D(self.f, z)) <= e:
                data = pd.DataFrame(cache).T
                data.columns = ['a', 'f(a)', 'b', 'f(b)', 'z', "D(f(z))", 'f(z)']
                self.data = data
                self.res = [round(z, prec), round(self.f(z), prec)]
                break
            else:
                if D(self.f, z) < 0:
                    a = z
                else:
                    b = z
            i += 1

    def rep(self):
        return print(
            f'Минимум функции f(x) = {self.func}, равный {round(self.res[1], self.prec)}  достигается при x = {round(self.res[0], self.prec)} с точностью e = {self.e} на шаге {len(self.data)}')

    def vis_steps(self, update=0.5):
        df, opt = self.data, self.res
        f, a, b = self.f, self.a, self.b
        prec = self.prec

        plt_x = np.linspace(a, b, round((b - a) * 250))
        plt_data = np.array([plt_x, [f(x) for x in plt_x]])
        plt_data_der = np.array([plt_x, [D(f, x) for x in plt_x]])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 9))

        for i in range(5):
            display(fig)
            ax1.cla()
            clear_output(wait=True)
            plt.pause(0.5)

        for i in range(len(df)):

            ax1.axhline(y=0, color='k', linewidth=1)
            ax1.axvline(x=0, color='k', linewidth=1)

            ax2.axhline(y=0, color='k', linewidth=1)
            ax2.axvline(x=0, color='k', linewidth=1)

            ax1.plot(plt_data_der[0], plt_data_der[1], c='blue', label="D(f(x))")
            ax2.plot(plt_data[0], plt_data[1], c='blue', label='f(x)')

            ax1.scatter(0, 0, c='white', label=f'step {i + 1}')

            if i != len(df) - 1:
                ax1.plot([df['a'][i], df['b'][i]], [D(f, df['a'][i]), D(f, df['b'][i])], 'ro-',
                         label=f'D(f(z)) = {round(df["D(f(z))"][i], prec)}')
                ax2.scatter(df['z'][i], df['f(z)'][i], label=f'f(z) = {round(df["f(z)"][i], prec)}')

                ax1.scatter(0, 0, c='white', label=f'z = {round(df["z"][i], prec)}')
                ax2.scatter(0, 0, c='white', label=f'z = {round(df["z"][i], prec)}')
            else:
                ax1.plot([df['a'][i], df['b'][i]], [D(f, df['a'][i]), D(f, df['b'][i])], 'ro-',
                         label=f'D(f(x*)) = {round(df["D(f(z))"][i], prec)}')
                ax2.scatter(df['z'][i], df['f(z)'][i], label=f'f(x*) = {round(df["f(z)"][i], prec)}')

                ax1.scatter(0, 0, c='white', label=f'x* = {round(df["z"][i], prec)}')
                ax2.scatter(0, 0, c='white', label=f'x* = {round(df["z"][i], prec)}')

            ax1.legend(prop={'size': 14})
            ax2.legend(prop={'size': 14})
            display(fig)

            ax1.cla()
            ax2.cla()

            clear_output(wait=True)
            plt.pause(update)


class Newton_search:
    def __init__(self, f, a, b, e, prec=4):
        raw_f = inspect.getsource(f)
        self.func = raw_f[raw_f.find('return') + 6:].strip()
        self.a, self.b, self.e = a, b, e
        self.prec = prec
        self.f = f

    def D(self, f, x):
        return derivative(f, x, dx=1e-6)

    def D2(self, f, x):
        return derivative(f, x, n=2, dx=1e-6)

    def fit(self):
        a, b, e, f = self.a, self.b, self.e, self.f
        prec = self.prec

        cache = {}
        i = 0
        z = b
        while 1:
            z = z - (D(self.f, z) / D2(self.f, z))
            cache[i] = [z, self.f(z), D(self.f, z), D2(self.f, z)]
            if abs(D(self.f, z)) <= e:
                data = pd.DataFrame(cache).T
                data.columns = ['z', 'f(z)', "D(f(z))", "D2(f(z))"]
                self.data = data
                self.res = [round(z, prec), round(self.f(z), prec)]
                break
            else:
                if D(self.f, z) < 0:
                    a = z
                else:
                    b = z
            i += 1

    def rep(self):
        return print(
            f'Минимум функции f(x) = {self.func}, равный {round(self.res[1], self.prec)}  достигается при x = {round(self.res[0], self.prec)} с точностью e = {self.e} на шаге {len(self.data)}')

    def vis_steps(self, update=0.5):
        df, opt = self.data, self.res
        f, a, b = self.f, self.a, self.b
        prec = self.prec

        plt_x = np.linspace(a, b, round((b - a) * 250))
        plt_data = np.array([plt_x, [f(x) for x in plt_x]])
        plt_data_der = np.array([plt_x, [D(f, x) for x in plt_x]])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 9))

        for i in range(5):
            display(fig)
            ax1.cla()
            clear_output(wait=True)
            plt.pause(0.5)

        for i in range(len(df)):

            ax2.set_ylim(min(plt_data[1]) - (max(plt_data[1]) - min(plt_data[1])) / 10, max(plt_data[1]))
            ax1.set_ylim(min(plt_data_der[1]), max(plt_data_der[1]))

            ax1.axhline(y=0, color='k', linewidth=1)
            ax1.axvline(x=0, color='k', linewidth=1)

            ax2.axhline(y=0, color='k', linewidth=1)
            ax2.axvline(x=0, color='k', linewidth=1)

            ax1.plot(plt_data_der[0], plt_data_der[1], c='blue', label="D(f(x))")
            ax2.plot(plt_data[0], plt_data[1], c='blue', label='f(x)')

            ax1.scatter(0, 0, c='white', label=f'step {i + 1}')

            sz = np.array([a, b])
            sy = df["D(f(z))"][i] * (sz - df['z'][i]) + df['f(z)'][i]
            dsy = df["D2(f(z))"][i] * (sz - df['z'][i]) + df['D(f(z))'][i]

            if i != len(df) - 1:
                ax1.scatter(0, 0, c='white', label=f'z = {round(df["z"][i], prec)}')
                ax2.scatter(0, 0, c='white', label=f'z = {round(df["z"][i], prec)}')

                ax1.scatter(df['z'][i], df['D(f(z))'][i], c='red', label=f'D(f(z)) = {round(df["D(f(z))"][i], prec)}')
                ax2.plot(sz, sy, c='red', label=f'D(f(z)) = {round(df["D(f(z))"][i], prec)}')
                ax1.plot(sz, dsy, c='red')
                ax2.scatter(df['z'][i], df['f(z)'][i], label=f'f(z) = {round(df["f(z)"][i], prec)}')
            else:
                ax1.scatter(0, 0, c='white', label=f'x* = {round(df["z"][i], prec)}')
                ax2.scatter(0, 0, c='white', label=f'x* = {round(df["z"][i], prec)}')

                ax1.scatter(df['z'][i], df['D(f(z))'][i], c='red', label=f'D(f(x*)) = {round(df["D(f(z))"][i], prec)}')
                ax2.plot(sz, sy, c='red', label=f'D(f(x*)) = {round(df["D(f(z))"][i], prec)}')
                ax1.plot(sz, dsy, c='red')
                ax2.scatter(df['z'][i], df['f(z)'][i], label=f'f(x*) = {round(df["f(z)"][i], prec)}')

            ax1.legend(prop={'size': 14})
            ax2.legend(prop={'size': 14})
            display(fig)

            ax1.cla()
            ax2.cla()

            clear_output(wait=True)
            plt.pause(update)

