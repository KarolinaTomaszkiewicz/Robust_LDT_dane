import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

def generate_data(t, A, sigma, omega, noise=0.0, n_outliers=0, random_state=0):
    y = A * np.exp(-sigma * t) * np.sin(omega * t)
    rnd = np.random.RandomState(random_state)
    error = noise * rnd.randn(t.size)
    outliers = rnd.randint(0, t.size, n_outliers)
    error[outliers] *=20
    return y + error

A = 2
sigma = 0.2
omega = 0.1 * 2 * np.pi
x_true = np.array([A, sigma, omega])
noise = 0.02
t_min = 0
t_max = 45
n_outliers = 2

param=["A=", A, "sigma=", sigma, "noise=", noise, "t_min=", t_min,"t_max=", t_max, "n_outliers=", n_outliers]
with open('parametry.txt', 'w') as f:
    for i in range(0, len(param),2):
        f.write(str(param[i])+str(param[i+1])+'\n')
    f.close



t_train = np.linspace(t_min, t_max, 30)
y_train = generate_data(t_train, A, sigma, omega, noise=noise, n_outliers=n_outliers)
x0 = np.ones(3)

def fun(x, t, y):
    return x[0] * np.exp(-x[1] * t) * np.sin(x[2] * t) - y

res_lsq = least_squares(fun, x0, args=(t_train, y_train))
print(f' res_lsq solution -> A:{res_lsq.x[0]:.2f} True A:{A}; simgma {res_lsq.x[1]:.2f}: True sigma {sigma}; omega {res_lsq.x[2]:.2f}: True omega{omega:.2f}')
res_robust = least_squares(fun, x0, loss='soft_l1', f_scale=0.1, args=(t_train, y_train))
print(f' res_robust solution -> A:{res_robust.x[0]:.2f}: True A:{A}; simgma {res_robust.x[1]:.2f}: True sigma {sigma}; omega {res_robust.x[2]:.2f}: True omega{omega:.2f}')
t_test = np.linspace(t_min, t_max, 300)
y_test = generate_data(t_test, A, sigma, omega)
y_lsq = generate_data(t_test, *res_lsq.x)
y_robust = generate_data(t_test, *res_robust.x)

zipped = zip(t_train, y_train)
dane=list(zipped)

with open('dane.txt', 'w') as f:
    for item in dane:
        item = str(item)
        item = item[1:]
        item = item[:-1]
        item = item.replace(',', '')
        f.write(item + '\n')

plt.plot(t_train, y_train, 'o', label='data')
plt.plot(t_test, y_test, label='true')
plt.plot(t_test, y_lsq, label='lsq')
plt.plot(t_test, y_robust, label='robust lsq')
plt.grid(color='black', linestyle='--', linewidth=0.5, markevery=int)
plt.xlabel('$t$')
plt.ylabel('$y$')
plt.title(r'$ f(t;A, \sigma, \omega ) = Ae^{- \sigma t} sin (\omega t)$')
plt.legend()
plt.savefig('wykres.png')
plt.show()