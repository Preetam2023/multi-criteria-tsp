import numpy as np


# ─────────────────────────────────────────────────────────────
# Pareto dominance
# ─────────────────────────────────────────────────────────────

def dominates(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    return np.all(a <= b) and np.any(a < b)


# ─────────────────────────────────────────────────────────────
# Pareto Archive
# ─────────────────────────────────────────────────────────────

class ParetoArchive:

    def __init__(self, max_size=200):

        self.solutions = []
        self.max_size = max_size


    # ---------------------------------------------------------
    # Add ONE solution safely
    # ---------------------------------------------------------

    def add(self, route, objectives):

        objectives = np.asarray(objectives)

        # -----------------------------------------------------
        # Duplicate prevention
        # -----------------------------------------------------

        for _, obj in self.solutions:

            if np.allclose(
                obj,
                objectives,
                atol=1e-6
            ):
                return False

        # -----------------------------------------------------
        # Check domination
        # -----------------------------------------------------

        dominated = False

        new_solutions = []

        for r, obj in self.solutions:

            # Existing dominates new
            if dominates(obj, objectives):
                dominated = True
                break

            # New dominates existing
            elif dominates(objectives, obj):
                continue

            else:
                new_solutions.append((r, obj))

        if dominated:
            return False

        # -----------------------------------------------------
        # Add new solution
        # -----------------------------------------------------

        new_solutions.append((
            route.copy(),
            objectives.copy()
        ))

        self.solutions = new_solutions

        # -----------------------------------------------------
        # Prune if needed
        # -----------------------------------------------------

        if len(self.solutions) > self.max_size:
            self._prune_crowding()

        return True


    # ---------------------------------------------------------
    # Batch update
    # ---------------------------------------------------------

    def update(self, pop, obj_costs):

        added = False

        for route, obj in zip(pop, obj_costs):

            was_added = self.add(route, obj)

            if was_added:
                added = True

        return added


    # ---------------------------------------------------------
    # Get objective matrix
    # ---------------------------------------------------------

    def get_objectives(self):

        if not self.solutions:
            return np.empty((0, 3))

        return np.array([
            obj for _, obj in self.solutions
        ])


    # ---------------------------------------------------------
    # Archive size
    # ---------------------------------------------------------

    def __len__(self):
        return len(self.solutions)


    # ---------------------------------------------------------
    # Crowding pruning
    # ---------------------------------------------------------

    def _prune_crowding(self):

        while len(self.solutions) > self.max_size:

            objs = self.get_objectives()

            cd = self._crowding_distance(objs)

            remove_idx = int(np.argmin(cd))

            self.solutions.pop(remove_idx)


    # ---------------------------------------------------------
    # Crowding distance
    # ---------------------------------------------------------

    def _crowding_distance(self, objs):

        n, m = objs.shape

        if n <= 2:
            return np.full(n, np.inf)

        cd = np.zeros(n)

        for k in range(m):

            idx = np.argsort(objs[:, k])

            cd[idx[0]] = np.inf
            cd[idx[-1]] = np.inf

            span = (
                objs[idx[-1], k]
                - objs[idx[0], k]
            )

            if span > 1e-10:

                for i in range(1, n - 1):

                    cd[idx[i]] += (
                        objs[idx[i + 1], k]
                        - objs[idx[i - 1], k]
                    ) / span

        return cd