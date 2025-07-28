import os
import numpy as np
import sys
import time
from tqdm import tqdm
from scipy import stats
import dill

from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.qpy import load as qpy_load
from qiskit.quantum_info import Statevector
from qiskit_algorithms import EstimationProblem
from qiskit_aer import AerSimulator
from qiskit_algorithms import IterativeAmplitudeEstimation

from qiskit_ibm_runtime import SamplerV2 as Sampler

# Use AerSimulator as the backend
backend = AerSimulator()

# Initialize Sampler
sampler = Sampler(backend)



# Function to load circuits from QPY files
def load_circuits(folder_path):
    with open(os.path.join(folder_path, 'transpiled_circuit_A.qpy'), 'rb') as f:
        A = qpy_load(f)[0]
    with open(os.path.join(folder_path, 'transpiled_circuit_Q.qpy'), 'rb') as f:
        Q = qpy_load(f)[0]
    return A, Q



def run_naive(A, target_epsilon=1e-3, nshots=10000, alpha=0.05, max_iterations=10 ** 3):
    a, b = 0.5, 0.5
    total_shots = 0
    start_time = time.time()
    
    for iteration in range(max_iterations):
        # Create a copy of A and add measurement
        A_measured = A.copy()
        c0 = ClassicalRegister(1, 'c0')
        A_measured.add_register(c0)
        A_measured.measure(0, c0[0])

        try:
            # Run circuit A_measured, perform nshots measurements
            job = sampler.run([A_measured], shots=nshots)
            result = job.result()
            
            # Get counts
            counts = result[0].data.c0.get_counts()
        except Exception as exc:
            print(f"Error during measurement: {exc}")
            print(f"Circuit details: {A_measured}")
            print(f"Sampler details: {sampler}")
            raise Exception("Measurement failed to complete successfully.") from exc
        
        total_shots += nshots

        if '1' not in counts:
            counts['1'] = 0
        if '0' not in counts:
            counts['0'] = 0
        
        # Update posterior distribution
        a += counts['1']
        b += counts['0']
        
        # Calculate 1-alpha confidence interval
        interval = stats.beta.interval(1-alpha, a, b)
        interval_half_length = (interval[1] - interval[0]) / 2
        
        if interval_half_length <= target_epsilon:
            end_time = time.time()
            max_num_Q = 0
            return total_shots, end_time - start_time, interval, max_num_Q
    
    # If maximum iterations reached without meeting the condition
    end_time = time.time()
    raise ValueError(f"Maximum iterations ({max_iterations}) reached without meeting the target epsilon. Last interval: {interval}")



def run_iae(problem, use_bayesian=False):
    iae = IterativeAmplitudeEstimation(
        epsilon_target=0.001,
        alpha=0.05,
        sampler=sampler,
        min_ratio=2
    )
    start_time = time.time()
    result = iae.estimate(problem, bayes=use_bayesian)
    end_time = time.time()
    
    running_time = end_time - start_time
    sample_complexity = result.num_oracle_queries
    interval = result.estimate_intervals[-1]
    max_num_Q = result.powers[-1]
    
    return sample_complexity, running_time, interval, max_num_Q



def run_algo(folder_path, num_iterations):
    A, Q = load_circuits(folder_path)
    
    # Create EstimationProblem for IAE methods
    problem = EstimationProblem(
        state_preparation=A,
        grover_operator=Q,
        objective_qubits=[0],
    )
    
    # Run each method and calculate averages
    methods = ['Naive', 'IAE without Bayesian', 'IAE with Bayesian']
    method_funcs = [run_naive, lambda prob: run_iae(prob, use_bayesian=False), lambda prob: run_iae(prob, use_bayesian=True)]
    
    folder_results = {}
    for method, func in zip(methods, method_funcs):
        print(f"  Running {method}")
        sample_complexities = []
        running_times = []
        intervals = []
        max_num_Qs = []
        
        progress_bar = tqdm(range(num_iterations), desc=f"  {method} Progress", leave=True)
        for _ in progress_bar:
            if method == 'Naive':
                sc, rt, itv, mnQ = func(A)  # Pass circuit A for Naive method
            else:
                sc, rt, itv, mnQ = func(problem)  # Pass problem for IAE methods
            sample_complexities.append(sc)
            running_times.append(rt)
            intervals.append(itv)
            max_num_Qs.append(mnQ)
        progress_bar.close()
        
        folder_results[method] = {
            'avg_sample_complexity': np.mean(sample_complexities),
            'std_err_sample_complexity': np.std(sample_complexities, ddof=1) / np.sqrt(len(sample_complexities)),
            'avg_running_time': np.mean(running_times),
            'std_err_running_time': np.std(running_times, ddof=1) / np.sqrt(len(running_times)),
            'avg_interval_center': np.mean([(itv[1]+itv[0])/2 for itv in intervals]),
            'std_err_interval_center': np.std([(itv[1]+itv[0])/2 for itv in intervals], ddof=1) / np.sqrt(len(intervals)),
            'avg_interval_half_length': np.mean([(itv[1]-itv[0])/2 for itv in intervals]),
            'std_err_interval_half_length': np.std([(itv[1]-itv[0])/2 for itv in intervals], ddof=1) / np.sqrt(len(intervals)),
            'avg_max_num_Q': np.mean(max_num_Qs),
            'std_err_max_num_Q': np.std(max_num_Qs, ddof=1) / np.sqrt(len(max_num_Qs))
        }
        
        print(f"  {method} Sample Complexity: {folder_results[method]['avg_sample_complexity']:.2f} ± {folder_results[method]['std_err_sample_complexity']:.2f}")
    
    return folder_results

if __name__ == "__main__":
    # Get the folder path and number of iterations from the command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python run_algo.py <folder_path> <num_iterations>")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    num_iterations = int(sys.argv[2])
    
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory")
        sys.exit(1)
    
    # Process the folder with the specified number of iterations
    results = run_algo(folder_path, num_iterations)
    
    # Save the results (optional)
    # output_dir = os.path.join(os.getcwd(), f'results_all_str')
    # os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(f'results_{os.path.basename(folder_path)}.pkl')
    with open(output_file, 'wb') as f:
        dill.dump(results, f)
    print(f"Results saved to {output_file}")