import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data_dir = '/mnt/extra/University/10 - Complex Systems/Project/data'
images_dir = '/mnt/extra/University/10 - Complex Systems/Project/images'


def line_model(x, a, b):
    return a * x + b


def fit_model(x, tau, s_0, A):
    return A - tau * x - np.exp(x) / s_0


def bin_s(name, N, gamma, samples, mid_value, step_factor, start_index, end_index):
    data = np.load(f'{data_dir}/{name}_{samples}_raw_data.npy', allow_pickle=True).tolist()
    S_s = data['S_s']
    G_s = data['G_s']
    T_s = data['T_s']

    min_num, max_num = np.min(S_s), np.max(S_s)
    bins = np.concatenate((np.arange(np.abs(min_num - 0.5), mid_value + 0.5, 1),
                           np.exp(np.arange(np.log(mid_value + 0.5), np.log(max_num + 0.5), np.log(step_factor)))))
    fres, edges = np.histogram(S_s, bins=bins, density=True)
    x_s = (edges[1:] + edges[:-1]) / 2

    params, covs = curve_fit(line_model, np.log(x_s[start_index:end_index]).astype(np.float64),
                             np.log(fres[start_index:end_index]).astype(np.float64))
    errs = np.sqrt(np.diag(covs))
    fit = np.exp(line_model(np.log(x_s), *params))
    print(f'tau = {params[0]} +- {errs[0]}')
    print(f'S_0 = {params[1]} +- {errs[1]}')

    plt.figure(figsize=(8, 5))
    plt.plot(x_s, fres, linestyle='', marker='o', markersize=4, label=f'$L = {gamma}$')
    # plt.plot(x_s, fit, linestyle='--')
    plt.plot(x_s[start_index:end_index], fit[start_index:end_index], linestyle='--')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$s$')
    plt.ylabel('$p(s)$')
    plt.legend()
    # plt.savefig(f'{model}_l_{length}_p_s.jpg')
    plt.show()

    data = {
        'x_s': x_s,
        'fres': fres,
        'params': params,
        'errs': errs
    }
    np.save(f'{data_dir}/{name}_{samples}_binned_data.npy', data)


def plot_s_s(names, gammas):
    plt.figure(figsize=(8, 6), dpi=300)
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive']
    markers = ['o', 'v', 's', '*', 'd', 'x', '<', 'x', '>']
    for i, name in enumerate(names):
        data = np.load(f'{data_dir}/{name}_binned_data.npy', allow_pickle=True).tolist()
        x_s = data['x_s']
        fres = data['fres']
        params = data['params']
        errs = data['errs']
        fit = np.exp(line_model(np.log(x_s), *params) + i)

        print(f'tau = {params[0]} +- {errs[0]}')

        plt.plot(x_s, np.exp(np.log(fres) + i), linestyle='', marker=markers[i], color=colors[i], markersize=4, label=f'$\gamma = {gammas[i]}$')
        plt.plot(x_s, fit, linestyle='--', linewidth=1.5, color=colors[i])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$s$')
    plt.ylabel('$p(s)$')
    plt.legend()
    plt.savefig(f'{images_dir}/Branching_p_s_s.jpg')
    plt.show()


def plot_s_t(name, N, gamma, samples, T_start, T_end):
    data = np.load(f'{data_dir}/{name}_{samples}_raw_data.npy', allow_pickle=True).tolist()
    S_s = data['S_s']
    T_s = data['T_s']

    indexes = np.where(np.logical_not(np.logical_or(T_s == 0, S_s == 0)))
    T_s = T_s[indexes]
    S_s = S_s[indexes]
    indexes = np.argsort(T_s)
    T_s = T_s[indexes]
    S_s = S_s[indexes]
    indexes = np.logical_and(T_s >= T_start, T_s <= T_end)
    T_s = T_s[indexes]
    S_s = S_s[indexes]

    params, covs = curve_fit(line_model, np.log(T_s[start_index:end_index]), np.log(S_s[start_index:end_index]))
    errs = np.sqrt(np.diag(covs))
    fit = np.exp(line_model(np.log(T_s), *params))
    print(f'tau = {params[0]} +- {errs[0]}')

    plt.scatter(T_s, S_s, s=1)
    plt.plot(T_s[start_index:end_index], fit[start_index:end_index], color='tab:orange')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()


N = 100000
gamma = 2.01
name = f'branching_{N}_{gamma}_new'
samples = 10000000

mid_value = 5
step_factor = 1.4
start_index = 10
end_index = None

gammas = [2.01, 2.2, 2.4, 2.6, 2.8, 3., 5.]

# # bin_s(name, N, gamma, samples, mid_value, step_factor, start_index, end_index)

names = [f'branching_100000_{gamma}_new_10000000' for gamma in gammas[:-1]] + [f'branching_100000_{gammas[-1]}_new_1000000']
# plot_s_s(names, gammas)

gamma = 2.8
N = 100000
samples = 10000000
T_start = 15
T_end = 100000
name = f'branching_{N}_{gamma}_new'
# plot_s_t(name, N, gamma, samples, T_start, T_end)
