import matplotlib.pyplot as plt
import numpy as np
from quantecon.distributions import BetaBinomial


class McCallModel:
    n, a, b = 12, 30, 30                        # default parameters
    q_default = BetaBinomial(n, a, b).pdf()
    w_min, w_max = 50, 150
    w_default = np.linspace(w_min, w_max, n+1)
    print(w_default)

    def __init__(self, c=25, β=0.99, w=w_default, q=q_default):

        self.c, self.β = c, β
        self.w, self.q = w, q

    def state_action_values(self, i, v):
        """
        The values of state-action pairs.
        """
        # Simplify names
        c, β, w, q = self.c, self.β, self.w, self.q
        # Evaluate value for each state-action pair
        # Consider action = accept or reject the current offer
        accept = w[i] / (1 - β)
        reject = c + β * np.sum(v * q)

        return np.array([accept, reject])

    def plot_value_function_seq(mcm, ax, num_plots=6):
        """
        Plot a sequence of value functions.

            * mcm is an instance of McCallModel
            * ax is an axes object that implements a plot method.

        """

        n = len(mcm.w)
        v = mcm.w / (1 - mcm.β)
        v_next = np.empty_like(v)
        for i in range(num_plots):
            ax.plot(mcm.w, v, '-', alpha=0.4, label=f"iterate {i}")
            # Update guess
            for i in range(n):
                v_next[i] = np.max(mcm.state_action_values(i, v))
            v[:] = v_next  # copy contents into v

        ax.legend(loc='lower right')

    def compute_reservation_wage(self, max_iter=500, tol=1e-6):

        # Simplify names
        c, β, w, q = mcm.c, mcm.β, mcm.w, mcm.q

        # == First compute the value function == #

        n = len(w)
        v = w / (1 - β)  # initial guess
        v_next = np.empty_like(v)
        i = 0
        error = tol + 1
        while i < max_iter and error > tol:

            for i in range(n):
                v_next[i] = np.max(mcm.state_action_values(i, v))

            error = np.max(np.abs(v_next - v))
            i += 1

            v[:] = v_next  # copy contents into v

        # == Now compute the reservation wage == #

        return (1 - β) * (c + β * np.sum(v * q))



mcm = McCallModel()

fig, ax = plt.subplots()
ax.set_xlabel('wage')
ax.set_ylabel('value')
McCallModel.plot_value_function_seq(mcm, ax)
print(mcm.compute_reservation_wage())
plt.show()
