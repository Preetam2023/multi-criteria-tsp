import numpy as np

def dominates(a, b):
    return np.all(a <= b) and np.any(a < b)

class ParetoArchive:
    def __init__(self, max_size=200):
        self.solutions = []
        self.max_size  = max_size

    def update(self, pop, obj_costs):
        added = False
        for pi, f in zip(pop, obj_costs):
            if any(dominates(g, f) for _, g in self.solutions):
                continue
            self.solutions = [(t, g) for t, g in self.solutions
                              if not dominates(f, g)]
            self.solutions.append((pi.copy(), f.copy()))
            added = True
        if len(self.solutions) > self.max_size:
            self._prune_crowding()
        return added

    def get_objectives(self):
        if not self.solutions:
            return np.empty((0, 3))
        return np.array([g for _, g in self.solutions])

    def __len__(self):
        return len(self.solutions)

    def _prune_crowding(self):
        while len(self.solutions) > self.max_size:
            objs = self.get_objectives()
            cd   = self._crowding_distance(objs)
            self.solutions.pop(int(np.argmin(cd)))

    def _crowding_distance(self, objs):
        n, m = objs.shape
        cd = np.zeros(n)
        for k in range(m):
            idx  = np.argsort(objs[:, k])
            cd[idx[0]] = cd[idx[-1]] = np.inf
            span = objs[idx[-1], k] - objs[idx[0], k]
            if span > 1e-10:
                for i in range(1, n - 1):
                    cd[idx[i]] += (objs[idx[i+1],k] - objs[idx[i-1],k]) / span
        return cd
