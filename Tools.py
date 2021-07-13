import numpy as np
import networkx as nx


def power_law_configuration_model(N, gamma, k_mean, dir, k_shift=0):
    k_range = range(N - 1, 0, -1)
    p_s = np.power(k_range, -gamma)
    p_cum_s = np.cumsum(p_s)[::-1]
    p_cum_s_norm = p_cum_s / p_cum_s[0]
    k_mean_s = np.cumsum(p_s * k_range)[::-1] / p_cum_s
    k_mean_index = np.argmin(np.abs(k_mean_s - (k_mean + k_shift)))

    k_s = np.zeros(N)
    r_s = np.random.rand(N) * p_cum_s_norm[k_mean_index]
    for i, r in enumerate(r_s):
        k_s[i] = np.where(r > p_cum_s_norm)[0][0]

    if np.sum(k_s) % 2 != 0:
        k_s[np.random.randint(0, len(k_s))] += 1
    graph = nx.configuration_model(k_s.astype(int))
    graph = nx.Graph(graph)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    print(f'k_mean = {np.mean(np.array(nx.degree(graph))[:, 1])}')
    nx.write_gpickle(graph, f'{dir}/configuration_model_{gamma}_{N}_{k_mean}.gpickle')


def power_law_HP_model(N, gamma, k_mean, gamma_shift, dir):
    alpha = 1 / ((gamma + gamma_shift) - 1)
    w_s = np.power(range(1, N + 1), -alpha)
    w_s /= np.sum(w_s)
    w_cum = np.cumsum(w_s)

    def w_rand(n):
        rand_s = np.random.rand(n)
        return np.searchsorted(w_cum, rand_s)

    edges = int(k_mean * N / 2)
    graph = nx.Graph()
    graph.add_nodes_from(range(N))
    needed_edges = edges

    while needed_edges > 0:
        print(f'\rNeed {needed_edges} more edges!', end='')
        nodes = w_rand(2 * needed_edges).reshape(needed_edges, 2)
        # diff_indexes = nodes[:, 0] != nodes[:, 1]
        # graph.add_edges_from(nodes[diff_indexes, :])
        graph.add_edges_from(nodes)
        needed_edges = edges - nx.number_of_edges(graph)
    print('')

    print(f'k_mean = {np.mean(np.array(nx.degree(graph))[:, 1])}')
    nx.write_gpickle(graph, f'{dir}/HP_model_{gamma}_{N}_{k_mean}.gpickle')
