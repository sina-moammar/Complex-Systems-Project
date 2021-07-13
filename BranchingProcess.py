import numpy as np


class BranchingProcess:
    def __init__(self, N, gamma):
        self.N = N
        self.gamma = gamma
        w_s = np.power(range(1, self.N + 1), -self.gamma)
        w_s_norm = w_s / np.sum(w_s * range(1, self.N + 1))
        w_s_ext = np.concatenate([[1 - np.sum(w_s_norm)], w_s_norm])
        self.w_cum = np.cumsum(w_s_ext)

    def new_branches_num(self, n):
        rands = np.random.rand(n)
        return np.searchsorted(self.w_cum, rands)

    def time_step(self):
        added = 1
        S = G = T = 0
        while added > 0:
            if S > 100 * self.N:
                return self.time_step()
            G += added
            T += 1
            new_branches = self.new_branches_num(added)
            S += np.count_nonzero(new_branches)
            added = np.sum(new_branches)

        return S, G, T
