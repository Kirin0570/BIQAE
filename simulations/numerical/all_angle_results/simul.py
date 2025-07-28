# version 2: save the detailed results of R repetitions

import os
import sys
import numpy as np
import dill

from qiskit.circuit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2
from qiskit_algorithms import EstimationProblem

from biae import BayesianIQAE as BIQAE



# Get theta_in_degrees from command line argument
theta_in_degrees = float(sys.argv[1])
p = np.sin(theta_in_degrees / 180*np.pi) ** 2

class BernoulliA(QuantumCircuit):
    """A circuit representing the Bernoulli A operator."""

    def __init__(self, probability):
        super().__init__(1)  # circuit on 1 qubit

        theta_p = 2 * np.arcsin(np.sqrt(probability))
        self.ry(theta_p, 0)


class BernoulliQ(QuantumCircuit):
    """A circuit representing the Bernoulli Q operator."""

    def __init__(self, probability):
        super().__init__(1)  # circuit on 1 qubit

        self._theta_p = 2 * np.arcsin(np.sqrt(probability))
        self.ry(2 * self._theta_p, 0)

    def power(self, k):
        # implement the efficient power of Q
        q_k = QuantumCircuit(1)
        q_k.ry(2 * k * self._theta_p, 0)
        return q_k


A = BernoulliA(p)
Q = BernoulliQ(p)

problem = EstimationProblem(
    state_preparation=A,  # A operator
    grover_operator=Q,  # Q operator
    objective_qubits=[0],  # the "good" state Psi1 is identified as measuring |1> in qubit 0
)

simulator_backend = AerSimulator()
sampler = SamplerV2(simulator_backend)

R = 1000  # Number of repetitions
epsilons = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
results_jeff = []   #Beta-IQAE
results_bayes = []  #Beta-BIQAE

for epsilon in epsilons:
    jeff_results_list = []
    bayes_results_list = []

    iae = BIQAE(
        epsilon_target=epsilon,  # target accuracy
        alpha=0.05,  
        sampler=sampler,
        min_ratio = 2
    )

    # Run R repetitions with bayes=False (Jeff)
    for _ in range(R):
        iae_result = iae.estimate(problem, bayes=False)
        jeff_results_list.append(iae_result)
    
    # Run R repetitions with bayes=True (Bayes)
    for _ in range(R):
        iae_result = iae.estimate(problem, bayes=True)
        bayes_results_list.append(iae_result)

    # Store the complete results for this epsilon
    results_jeff.append({
        'epsilon': epsilon,
        'results': jeff_results_list
    })
    
    results_bayes.append({
        'epsilon': epsilon,
        'results': bayes_results_list
    })

# Save data
data_to_save = {
    'results_bayes': results_bayes,
    'results_jeff': results_jeff,
    'theta': theta_in_degrees,
    'p': p
}

# Save pickle file with theta suffix
with open(f'results_{int(theta_in_degrees)}.pkl', 'wb') as f:
    dill.dump(data_to_save, f)
