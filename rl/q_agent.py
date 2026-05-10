import numpy as np

def extract_state(pop, fitness, obj_costs, g, G_max, stag_count,
                  theta1=0.1, theta2=0.3):
    sigma   = np.std(fitness)
    f_range = np.max(np.abs(fitness)) + 1e-10
    sigma_n = sigma / f_range
    div_bin  = 0 if sigma_n < theta1 else (2 if sigma_n > theta2 else 1)
    best_f   = np.min(fitness)
    best_bin = min(4, int(best_f * 10))
    impr_bin = 0 if stag_count == 0 else (2 if stag_count >= 5 else 1)
    mean_objs  = obj_costs.mean(axis=0)
    imbal_bin  = int(np.argmax(mean_objs))
    stag_bin   = 0 if stag_count == 0 else (2 if stag_count >= 5 else 1)
    ratio      = g / max(G_max, 1)
    phase_bin  = 0 if ratio < 0.3 else (2 if ratio >= 0.7 else 1)
    return (div_bin, best_bin, impr_bin, imbal_bin, stag_bin, phase_bin)

def compute_reward(f_prev, f_curr, sigma_prev, sigma_curr,
                   archive_grew, stag_count,
                   beta=0.3, rho=0.5, mu=0.5):
    r_fit    = (f_prev - f_curr) / abs(f_prev) if abs(f_prev) > 1e-10 else 0.0
    r_div    = beta if sigma_curr > sigma_prev else 0.0
    r_pareto = rho  if archive_grew else 0.0
    r_stag   = -mu  if stag_count >= 5 else 0.0
    return r_fit + r_div + r_pareto + r_stag

class QAgent:
    ACTION_NAMES = ['+δ w_D','-δ w_D','+δ w_T','-δ w_T','+δ w_C','-δ w_C']

    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1, delta=0.05):
        self.alpha=alpha; self.gamma=gamma; self.epsilon=epsilon; self.delta=delta
        self.Q={}; self.n_actions=6

    def _get_q(self, s):
        if s not in self.Q:
            self.Q[s] = np.zeros(self.n_actions)
        return self.Q[s]

    def select_action(self, s):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self._get_q(s)))

    def update(self, s, a, r, s_next):
        q = self._get_q(s)
        q[a] += self.alpha*(r + self.gamma*np.max(self._get_q(s_next)) - q[a])

    def apply_action(self, w, action):
        w = w.copy()
        w[action//2] += (1 if action%2==0 else -1) * self.delta
        w = np.clip(w, 0.05, 0.90)
        return w / w.sum()

    def get_q_stats(self):
        if not self.Q: return {'n_states_visited':0,'mean_q':0,'max_q':0,'min_q':0}
        all_q = np.vstack(list(self.Q.values()))
        return {'n_states_visited':len(self.Q),'mean_q':all_q.mean(),
                'max_q':all_q.max(),'min_q':all_q.min()}
