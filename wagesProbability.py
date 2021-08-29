import matplotlib.pyplot as plt
import numpy as np
from quantecon.distributions import BetaBinomial

# default parameters
# n- number of trials
n, a, b = 12, 30, 30
q_default = BetaBinomial(n, a, b).pdf()
w_min, w_max = 50, 150
w_default = np.linspace(w_min, w_max, n+1)
fig, ax = plt.subplots()
ax.plot(w_default, q_default, '-o', label='$q(w(i))$')
ax.set_xlabel('wages')
ax.set_ylabel('probabilities')

plt.show()



