# suggested values
alpha_val = 0.9
beta_val = 0.2
small_gamma_val = 0.1
lambda_0 = 1
lambda_1 = 5

n = 10
big_T = 100



# implemeting the transition matrix gamma 
import numpy as np

def create_transition_matrix(gamma, beta):
    Gamma = np.array([
            [(1-gamma)  ,0          ,gamma],
            [0          ,(1-gamma)  ,gamma],
            [(beta/2)   ,(beta/2)   ,(1-beta)]
            ], dtype=float)
    return Gamma

print(create_transition_matrix(small_gamma_val, beta_val))

# implemeting the conditional probability for Z_{t,i}
def p_z1_given_c(alpha, c):
    if c == 0:
        return (1-alpha)
    elif c == 1:
        return alpha 
    elif c == 2:
        return (0.5)
    else:
        raise ValueError("c must be 0, 1 or 2")

def simulate_c(T, gamma, beta, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    Gamma = create_transition_matrix(gamma, beta)

    C = np.empty(T, dtype=int)
    C[0] =  2 

    states = np.array([0, 1, 2], dtype=int)

    for t in range(1, T):
        C[t] = rng.choice(states, p=Gamma[C[t-1]])

    return C
