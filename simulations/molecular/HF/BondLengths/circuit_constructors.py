'''
NOTE :
The reversed in the Hadamard test is only used because of the taper mapping. For Jordon-Weigner and brayev-Kiteav we use the normal pauli string(NOT the reversed) 
''' 


from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2, TwoLocal, NLocal, RealAmplitudes, GroverOperator
from qiskit_nature.second_q.mappers import JordanWignerMapper, BravyiKitaevMapper
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock

def construct_ansatz(n, l, Str):
    if Str == 'EfficientSU2':
        qc = EfficientSU2(n, reps=l, entanglement='full', skip_unentangled_qubits=False, parameter_prefix='a')
    elif Str == 'Twolocal':
        qc = TwoLocal(n, ['ry','ry'], 'cx', 'full', reps=l, insert_barriers=False, parameter_prefix='a')
    elif Str == 'NLocal':
        qc = NLocal(n, ['ry','ry'], 'cx', 'full', reps=l, insert_barriers=False, parameter_prefix='a')    
    elif Str == 'RealAmplitudes':
        qc = RealAmplitudes(n, entanglement='full', reps=l, parameter_prefix='a')
    elif Str == "UCCSD":
        num_particles = (1, 1)
        num_spatial_orbitals = 5 
        mapper = BravyiKitaevMapper()
        init_state = HartreeFock(num_spatial_orbitals, num_particles, mapper)
        qc = UCCSD(
        num_spatial_orbitals, num_particles, mapper, initial_state=init_state)
    else:
        raise Exception("Enter correct input")
    
    return qc, qc.parameters


def construct_A(l, opt_params, pauli_string):
    # Prepare the ansatz
    nqubits = len(pauli_string)
    ansatz, _ = construct_ansatz(nqubits, l=l, Str="EfficientSU2")
    opt_ansatz = ansatz.assign_parameters(opt_params)
    
    # Compose the Hadamard test
    A = QuantumCircuit(nqubits + 1)

    A = A.compose(opt_ansatz, range(1, nqubits + 1))
    
    # Create U gate based on pauli_string
    U = QuantumCircuit(nqubits)
    for i, pauli in enumerate(reversed(str(pauli_string))):  
        if pauli == 'X':
            U.x(i)
        elif pauli == 'Y':
            U.y(i)
        elif pauli == 'Z':
            U.z(i)
    U_gate = U.to_gate(label="U gate")

    # Create Hadamard test circuit
    HT = QuantumCircuit(nqubits + 1)
    HT.h(0)
    HT = HT.compose(U_gate.control(1), list(range(0, nqubits + 1)))
    HT.h(0)

    # Compose final A circuit
    A = A.compose(HT, range(0, nqubits + 1))

    return A

# Example usage:
# A = construct_A(l=3, opt_params=res.x, pauli_string="XXYY")



def construct_Q(state_preparation):
    nqubits = state_preparation.num_qubits

    # Construct the oracle
    oracle = QuantumCircuit(nqubits)
    oracle.h(0)
    oracle.x(0)
    oracle.h(0)

    # Create the GroverOperator
    Q = GroverOperator(oracle, state_preparation)
    
    return Q

# Example usage:
# Q = construct_Q(state_preparation)