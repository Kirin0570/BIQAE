import dill
import math
import time
from scipy import stats
import os
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
import gc

from qiskit import ClassicalRegister
from qiskit.qpy import load as qpy_load
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler

# Constants
R = 200
backend = AerSimulator()
sampler = Sampler(backend)

# Function to run naive fixed shots
def run_naive_fixed_shots(A, n_shots=10000, alpha=0.05):
    a, b = 0.5, 0.5
    start_time = time.time()
    
    # Create a copy of A and add measurement
    A_measured = A.copy()
    c0 = ClassicalRegister(1, 'c0')
    A_measured.add_register(c0)
    A_measured.measure(0, c0[0])

    try:
        job = sampler.run([A_measured], shots=n_shots)
        result = job.result()
        counts = result[0].data.c0.get_counts()
    except Exception as exc:
        print(f"Error during measurement: {exc}")
        raise Exception("Measurement failed to complete successfully.") from exc

    if '1' not in counts:
        counts['1'] = 0
    if '0' not in counts:
        counts['0'] = 0

    a += counts['1']
    b += counts['0']
    interval = stats.beta.interval(1 - alpha, a, b)

    end_time = time.time()
    max_num_Q = 0
    return n_shots, end_time - start_time, interval, max_num_Q

# Function to get naive results with fixed shots
def get_res_naive_fixed_shots(A, n_shots, R):
    sample_complexities, running_times, intervals, max_num_Qs = [], [], [], []
    progress_bar = tqdm(range(R), desc="Progress", leave=True)

    for _ in progress_bar:
        sc, rt, itv, mnQ = run_naive_fixed_shots(A, n_shots)
        sample_complexities.append(sc)
        running_times.append(rt)
        intervals.append(itv)
        max_num_Qs.append(mnQ)
    progress_bar.close()

    res = {
        'avg_sample_complexity': np.mean(sample_complexities),
        'std_err_sample_complexity': np.std(sample_complexities, ddof=4) / np.sqrt(len(sample_complexities)),
        'avg_running_time': np.mean(running_times),
        'std_err_running_time': np.std(running_times, ddof=1) / np.sqrt(len(running_times)),
        'avg_interval_center': np.mean([(itv[1]+itv[0])/2 for itv in intervals]),
        'std_err_interval_center': np.std([(itv[1]+itv[0])/2 for itv in intervals], ddof=4) / np.sqrt(len(intervals)),
        'avg_interval_half_length': np.mean([(itv[1]-itv[0])/2 for itv in intervals]),
        'std_err_interval_half_length': np.std([(itv[1]-itv[0])/2 for itv in intervals], ddof=4) / np.sqrt(len(intervals)),
        'avg_max_num_Q': np.mean(max_num_Qs),
        'std_err_max_num_Q': np.std(max_num_Qs, ddof=1) / np.sqrt(len(max_num_Qs))
    }
    return res

# Function to process each file
def process_file(file):
    file_path = os.path.join('run_algo', file)
    key = file.split("_")[1].split(".")[0]

    with open(file_path, 'rb') as f:
        results_i = dill.load(f)

    n_shots = math.ceil(results_i['IAE with Bayesian']['avg_sample_complexity'])
    with open(os.path.join('all_str_A_Q', f'circuits/{key}', 'transpiled_circuit_A.qpy'), 'rb') as f:
        A = qpy_load(f)[0]
    
    new_naive = get_res_naive_fixed_shots(A, n_shots, R)
    results_i['Naive_fixed_shots'] = new_naive

    return key, results_i

# Main function using batch processing
if __name__ == "__main__":
    results = {}
    files = [file for file in os.listdir('run_algo') if file.startswith("results_") and file.endswith(".pkl")]
    
    # Define the batch size
    batch_size = 20

    for i in range(0, len(files), batch_size):
        batch = files[i:i + batch_size]
        
        # Use multiprocessing Pool to process the batch
        with Pool() as pool:
            processed_files = pool.map(process_file, batch)
        
        # Collect the results into the dictionary
        for key, results_i in processed_files:
            results[key] = results_i

        # Clear memory and run garbage collection
        processed_files = None
        gc.collect()

    # Save all results to a file
    with open("all_results.pkl", 'wb') as f:
        dill.dump(results, f, protocol=4)
