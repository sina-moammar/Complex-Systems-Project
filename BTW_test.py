import numpy as np
import networkx as nx
from BTWOnNetwork import BTWOnNetwork
import time


data_dir = '/mnt/extra/University/10 - Complex Systems/Project/data'
images_dir = '/mnt/extra/University/10 - Complex Systems/Project/images'


def save_avalanches_data(name, samples, S_s, A_s, G_s, T_s):
    data = {
        'S_s': S_s,
        'A_s': A_s,
        'G_s': G_s,
        'T_s': T_s,
    }
    np.save(f'{data_dir}/{name}_{samples}_raw_data.npy', data)


def collect_BTW_avalanches_data(name, graph, f, samples, uniform_threshold=None):
    model = BTWOnNetwork(graph, f, uniform_threshold)
    S_s = np.zeros(samples, dtype=np.uint32)
    A_s = np.zeros(samples, dtype=np.uint32)
    G_s = np.zeros(samples, dtype=np.uint32)
    T_s = np.zeros(samples, dtype=np.uint32)
    valid_index = int(1.75 * model.N)

    print(f'Going To Equilibrium:')
    for step in range(valid_index):
        print(f'\r\t{step + 1} from {valid_index}', end='')
        model.time_step()
    print('')

    print(f'Sampling:')
    for step in range(samples):
        print(f'\r\t{step + 1} from {samples}', end='')
        S_s[step], A_s[step], G_s[step], T_s[step] = model.time_step()
        if step % 1000000 == 0 and step != 0:
            save_avalanches_data(name, samples, S_s[:step], A_s[:step], G_s[:step], T_s[:step])
    print('')

    save_avalanches_data(name, samples, S_s, A_s, G_s, T_s)


start = time.time()
N = 100000
k_mean = 4
gamma = 2.2
f = 1e-2
graph = nx.read_gpickle(f'{data_dir}/HP_model_{gamma}_{N}_{k_mean}.gpickle')
name = f'{N}_{gamma}_{k_mean}_{f}_new'
samples = 10000000
collect_BTW_avalanches_data(name, graph, f, samples, uniform_threshold=2)
end = time.time()
print(f'Time: {end - start}')
