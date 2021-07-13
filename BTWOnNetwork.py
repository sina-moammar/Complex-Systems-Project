import numpy as np
import networkx as nx
import itertools


class BTWOnNetwork:
    def __init__(self, graph, f, uniform_threshold=None):
        self.graph = graph
        self.f = f
        self._grain = (1 - f)
        self.N = nx.number_of_nodes(self.graph)
        self._degrees = np.array(nx.degree(self.graph), dtype=np.uint32)[:, 1]
        self.threshold = self._degrees.copy()
        if uniform_threshold is not None:
            self.threshold[self._degrees > 1] = uniform_threshold
        self.h_s = np.floor(np.random.rand(self.N) * self._degrees)

    def time_step(self):
        changed_1 = []
        changed_2 = iter([])
        first_node = np.random.randint(0, self.N)
        while self._degrees[first_node] == 0:
            first_node = np.random.randint(0, self.N)
        changed_1.append(first_node)
        S = G = T = 0
        all_nodes = []
        is_first = True

        changed_length = 1
        while changed_length:
            changed_length = 0
            T += 1
            for changed in changed_1:
                all_nodes.append(changed)
                G += 1
                self.h_s[changed] += self._grain if not is_first else 1
                is_first = False
                if self.h_s[changed] >= self.threshold[changed]:
                    S += 1
                    self.h_s[changed] -= self.threshold[changed]
                    neighbors = nx.neighbors(self.graph, changed)
                    changed_2 = itertools.chain(changed_2, neighbors)
                    changed_length += 1

            changed_1, changed_2 = changed_2, iter([])

        all_unique_nodes = np.unique(all_nodes)
        A = len(all_unique_nodes)
        return S, A, G, T
