import numpy as np


def create_transition_matrix(gamma_param: float, beta_param: float) -> np.ndarray:
    """
    Creates a transition matrix for our C states.
    """
    Gamma = np.array([
            [(1-gamma_param)  ,0          ,gamma_param],
            [0          ,(1-gamma_param)  ,gamma_param],
            [(beta_param/2)   ,(beta_param/2)   ,(1-beta_param)]
            ], dtype=float)
    return Gamma

def p_z1_given_c(alpha_param: float, c: int) -> float:
    """
    Returns the probability of z=1 given c.
    """
    if c == 0:
        return (1-alpha_param)
    elif c == 1:
        return alpha_param 
    elif c == 2:
        return (0.5)
    else:
        raise ValueError("c must be 0, 1 or 2")
    
def simulate_c(T: int, gamma_param: float, beta_param: float, rng=None) -> np.ndarray:
    """
    Simulates the C states for T time steps given the gamma and beta parameters.
    """
    rng = np.random.default_rng() if rng is None else rng
    Gamma = create_transition_matrix(gamma_param, beta_param)
    #print(Gamma)

    C = np.empty(T, dtype=int)
    C[0] =  2  #P(C_1=2)=1, just the first state

    states = np.array([0, 1, 2], dtype=int)

    for t in range(1, T):
        C[t] = rng.choice(states, p=Gamma[C[t-1]])

    return C

def simulate_z(C: np.ndarray, n: int, alpha_param, rng=None):
    """
    Simulates the Z states for T times, for n neurons, given the C states and the alpha parameter.
    """
    rng = np.random.default_rng() if rng is None else rng
    T = len(C)

    Z = np.empty((T, n), dtype=int)

    for t in range(T):
        p = p_z1_given_c(alpha_param, C[t])
        Z[t] = rng.binomial(n=1, p=p, size=n)

    return Z

def simulate_x(Z: np.ndarray, lambda_0: int, lambda_1: int, rng=None):
    """
    Simulates the X states for T times, for n neurons, given the Z states and the lambda parameters.
    """
    rng = np.random.default_rng() if rng is None else rng
    lam = lambda_0 + (lambda_1 - lambda_0) * Z
    X = rng.poisson(lam=lam)
    return X

def simulate_hmm(T: int, n: int, alpha_param: float, beta_param: float, gamma_param: float, lambda_0:int, lambda_1:int, seed=None):
    """
    Simulates the whole HMM process for T time steps, n neurons, and given the parameters. Returns the C, Z and X states.
    """
    rng = np.random.default_rng(seed)

    c = simulate_c(T, gamma_param, beta_param, rng)
    z = simulate_z(c, n, alpha_param, rng)
    x = simulate_x(z, lambda_0, lambda_1, rng)

    return c, z, x 