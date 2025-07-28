import numpy as np
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool

def total_energy(bond_length):
    # Define the HF molecule at the specified bond length
    driver = PySCFDriver(
        atom=f"H 0 0 {-bond_length/2}; F 0 0 {bond_length/2}",
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM
    )
    
    molecule = driver.run()
    transformer = FreezeCoreTransformer()
    transformed_molecule = transformer.transform(molecule)

    nuclear_repulsion_energy = transformed_molecule.nuclear_repulsion_energy
    core_electron_energy = transformed_molecule.hamiltonian.constants['FreezeCoreTransformer']

    fermionic_op = transformed_molecule.hamiltonian.second_q_op()
    mapper = JordanWignerMapper()
    qubit_op = mapper.map(fermionic_op)
    
    hamiltonian_matrix = qubit_op.to_matrix()
    electronic_energy = np.min(np.linalg.eigvals(hamiltonian_matrix))
    
    return nuclear_repulsion_energy, electronic_energy, core_electron_energy

def calculate_energy_parallel(bond_length):
    try:
        return total_energy(bond_length)
    except Exception as e:
        print(f"Error calculating energy for bond length {bond_length}: {e}")
        return None, None, None

def generate_energy_curve_parallel(start, stop, spacing, num_workers):
    bond_lengths = np.arange(start, stop + spacing, spacing)
    total_energies = []
    nuclear_repulsion_energy_list = []
    electronic_energy_list = []
    frozen_electron_energy_list = []

    # Use multiprocessing Pool to parallelize the calculations
    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(calculate_energy_parallel, bond_lengths), total=len(bond_lengths), desc="Calculating energies"))

    for result in results:
        if result is not None:
            nuclear, electronic, frozen_electron_energy = result
            total_energies.append(nuclear + electronic + frozen_electron_energy)
            nuclear_repulsion_energy_list.append(nuclear)
            electronic_energy_list.append(electronic)
            frozen_electron_energy_list.append(frozen_electron_energy)

    return bond_lengths, np.real(total_energies), np.real(nuclear_repulsion_energy_list), np.real(electronic_energy_list), np.real(frozen_electron_energy_list)

# Parameters
start = 0.6
stop = 3.5
spacing = 0.01
num_points = int((stop - start) / spacing) + 1
num_workers = 10  # Adjust this number based on available cores on HTC

# Generate the energy curve in parallel
bond_lengths, total_energies, nuclear_repulsion_energy_list, electronic_energy_list, frozen_electron_energy_list = generate_energy_curve_parallel(start, stop, spacing, num_workers)

# # Plotting
# plt.figure(figsize=(8, 6))
# plt.plot(bond_lengths, total_energies, label='Total Energy', color='blue')
# plt.title("Total Energy vs Bond Length for H2 Molecule")
# plt.xlabel("Bond Length (Å)")
# plt.ylabel("Total Energy (Hartree)")
# plt.grid(True)
# plt.legend()
# plt.savefig('exact_curve.png', dpi=300)

# Save to CSV
df = pd.DataFrame({
    "BL": bond_lengths,
    "NRE": np.real(nuclear_repulsion_energy_list) + np.real(frozen_electron_energy_list),
    "EE": electronic_energy_list
})
df.to_csv('exact_curve.csv', index=False)
