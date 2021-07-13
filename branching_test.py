import numpy as np
from BranchingProcess import BranchingProcess
import time


def line_model(x, a, b):
    return a * x + b


data_dir = '/mnt/extra/University/10 - Complex Systems/Project/data'
images_dir = '/mnt/extra/University/10 - Complex Systems/Project/images'


def save_avalanches_data(name, samples, S_s, G_s, T_s):
    data = {
        'S_s': S_s,
        'G_s': G_s,
        'T_s': T_s,
    }
    np.save(f'{data_dir}/branching_{name}_{samples}_raw_data.npy', data)


def collect_branching_avalanches_data(name, N, gamma, samples):
    model = BranchingProcess(N, gamma)
    S_s = np.zeros(samples, dtype=np.uint32)
    G_s = np.zeros(samples, dtype=np.uint32)
    T_s = np.zeros(samples, dtype=np.uint32)

    print(f'Sampling:')
    for step in range(samples):
        print(f'\r\t{step + 1} from {samples}', end='')
        S_s[step], G_s[step], T_s[step] = model.time_step()
    print('')

    save_avalanches_data(name, samples, S_s, G_s, T_s)

start = time.time()
N = 100000
gamma = 3.
name = f'{N}_{gamma}_new'
samples = 10000000
collect_branching_avalanches_data(name, N, gamma, samples)
end = time.time()
print(f'Time: {end - start}')
