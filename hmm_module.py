import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import poisson


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

def simulate_z(C: np.ndarray, n: int, alpha_param, rng=None) -> np.ndarray:
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

def simulate_x(Z: np.ndarray, lambda_0: int, lambda_1: int, rng=None) -> np.ndarray:
    """
    Simulates the X states for T times, for n neurons, given the Z states and the lambda parameters.
    """
    rng = np.random.default_rng() if rng is None else rng
    lam = lambda_0 + (lambda_1 - lambda_0) * Z
    X = rng.poisson(lam=lam)
    return X

def simulate_hmm(T: int, n: int, alpha_param: float, beta_param: float, gamma_param: float, lambda_0:int, lambda_1:int, seed=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulates the whole HMM process for T time steps, n neurons, and given the parameters. Returns the C, Z and X states.
    """
    rng = np.random.default_rng(seed)

    c = simulate_c(T, gamma_param, beta_param, rng)
    z = simulate_z(c, n, alpha_param, rng)
    x = simulate_x(z, lambda_0, lambda_1, rng)

    return c, z, x 


def plot_single(X: np.ndarray) -> None:
    """
    Plots the spike count for a single neuron over time.
    """
    t_vals = np.arange(1, X.shape[0]+1)

    plt.figure(figsize=(12,4))
    plt.plot(t_vals, X[:, 0], marker="o", markersize=4)
    plt.xlabel("t")
    plt.ylabel(r"$X_{t,1}$")
    plt.grid(alpha=0.6)
    plt.title("spikes count over time for neuron 1")
    plt.show()

def plot_freq(X: np.ndarray) -> None:
    """
    Plots the frequency of spike counts across all neurons and time points.
    (This was to show the 2 different distributions)
    """
    counts = {}
    for i in X:
        for j in i:
            if j in counts:
                counts[j] += 1
            else:
                counts[j] = 1
    plt.figure(figsize=(12,4))
    plt.bar(counts.keys(), counts.values())
    plt.xlabel("spikes count")
    plt.ylabel("frequency")
    plt.title("frequency of spikes count across all neurons and time points")
    plt.grid(alpha=0.6)
    plt.show()

def plot_all(X: np.ndarray) -> None:
    """
    Plots the spike count for all neurons over time in subplots."""
    t_vals = np.arange(1, X.shape[0]+1)
    n = X.shape[1]
    fig, axes = plt.subplots(n, 1, figsize=(12, 1.4*n))
    for i in range(n):
        axes[i].plot(t_vals, X[:, i], marker="o", markersize=4)
        axes[i].set_xlabel("t")
        axes[i].set_ylabel(rf"$X_{{t,{i+1}}}$")
        axes[i].grid(alpha=0.6)
    plt.suptitle("spikes count over time for all neurons", y=0.99)
    plt.tight_layout()
    plt.show()

def plot_mean(X: np.ndarray) -> None:
    """
    Plots the mean spike count over all neurons at each time point, with a smoothed version using a Gaussian filter.
    """
    t_vals = np.arange(1, X.shape[0] + 1)
    mean_X = X.mean(axis=1)
    smooth_mean = gaussian_filter1d(mean_X, sigma=5)

    plt.figure(figsize=(12, 3))
    plt.plot(t_vals, mean_X, marker="o", markersize=4)
    plt.plot(t_vals, smooth_mean, color="r")
    plt.xlabel("t")
    plt.ylabel(r"$\bar{X}_t$")
    plt.grid(alpha=0.6)
    plt.title("mean spikes count over time")
    plt.show()

def make_datasets(M: int, t_index: int, T: int, n: int, alpha_param, beta_param, gamma_param, lambda_0, lambda_1, seed=123) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulates M independent HMM sequences, where we save the features (spike counts) and labels (hidden states) at specific time indexes, as well as the full sequences of C, Z and X for all M experiments.
    """
    rng = np.random.default_rng(seed)

    x_features = np.empty((M, n), dtype=float)      # one row per experiment 
    y_labels = np.empty(M, dtype=int)              # class label C_t in {0, 1, 2}

    X_full = np.empty((M, T, n), dtype=float)
    C_full = np.empty((M, T), dtype=int)
    Z_full = np.empty((M, T, n), dtype=int)

    for i in range(M):
        # simulate a new HMM sequence
        C, Z, X = simulate_hmm(T, n, alpha_param, beta_param, gamma_param, lambda_0, lambda_1, seed=int(rng.integers(0, 10000000)))
        x_features[i] = X[t_index, :] # features are the spike counts of all neurons at time t_index
        y_labels[i] = C[t_index]     # label is the hidden state at time t_index

        X_full[i] = X
        C_full[i] = C
        Z_full[i] = Z

    return x_features, y_labels, X_full, C_full, Z_full

def emission_matrix(X: np.ndarray, alpha_param : float, lambda0: int, lambda1: int) -> np.ndarray:
    """
    We compute the emission matrix B, to reduce the problem to a HMM
    """
    T, n = X.shape
    pis = np.array([1-alpha_param, alpha_param, 0.5])

    p0 = poisson.pmf(X, mu=lambda0)
    p1 = poisson.pmf(X, mu=lambda1)

    B = np.zeros((T, 3), dtype=float)
    for c in range(3):
        pi = pis[c]
        mix = (1-pi)*p0 + pi*p1    
        B[:, c] = np.prod(mix, axis=1)
    return B

def forward_C(B: np.ndarray, Gamma: np.ndarray, pi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Accumulating probabilities of the hidden states over time, given the emission probabilities and the transition matrix. We also keep track of the scaling factors to avoid numerical underflow.
    """
    T = B.shape[0]
    scale = np.zeros(T, dtype=float)
    alpha = np.zeros((T, 3), dtype=float)
    alpha[0] = B[0]*pi
    scale[0] = alpha[0].sum()
    alpha[0] = alpha[0]/scale[0]

    for t in range(1, T):
        alpha[t] = (alpha[t-1] @ Gamma) * B[t]
        scale[t] = alpha[t].sum()
        alpha[t] = alpha[t]/scale[t]
    
    return alpha, scale

def backward_C(B: np.ndarray, scale:np.ndarray, Gamma: np.ndarray) -> np.ndarray:
    """
    Computing the backward probabilities of the hidden states over time, given the emission probabilities, the transition matrix and the scaling factors from the forward pass.
    """
    T = B.shape[0]
    beta = np.zeros((T,3), dtype=float)
    beta[-1] = 1.0

    for t in range(T-2, -1, -1):
        beta[t] = Gamma @ (B[t+1] * beta[t+1])
        beta[t] = beta[t]/scale[t+1]
    return beta

def compute_smoothed_prob(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Combining the forward and backward probabilities to compute the smoothed probabilities of the hidden states at each time point, given all the observations.
    """
    smoothed = alpha * beta
    smoothed = smoothed / smoothed.sum(axis=1, keepdims=True)  # issues with floating point numbers
    return smoothed

def posterior_Z(X: np.ndarray, qC: np.ndarray, alpha_param: float, lambda0: int, lambda1: int) -> np.ndarray:
    """
    Computing the posterior probability of Z=1 given the observations X and the smoothed probabilities of C, using Bayes rule and marginalizing over C.
    """
    T, n = X.shape
    pis = np.array([1- alpha_param, alpha_param, 0.5])

    p0 = poisson.pmf(X, mu=lambda0)
    p1 = poisson.pmf(X, mu=lambda1)

    qZ1_given_c = np.zeros((T, n, 3), dtype=float)

    for c in range(3):
        pi_c = pis[c]
        demoninator = (1 - pi_c)* p0 + pi_c*p1
        qZ1_given_c[:, :, c] = (pi_c*p1) / demoninator

    #marginalize over C_t using qC(t,c)
    qZ1 = np.zeros((T, n), dtype=float)

    for t in range(T):
        for i in range(n):
            # we have the contribution from C_t={0, 1, 2}
            qZ1[t, i ] = (qZ1_given_c[t,i,0] * qC[t,0] + qZ1_given_c[t,i,1] * qC[t,1] + qZ1_given_c[t,i,2] * qC[t,2])
    return qZ1

def hmm_pipeline(X: np.ndarray, alpha_param: float, beta_param: float, gamma_param: float, lambda0: int, lambda1:int ) -> dict[str, np.ndarray]:
    """
    Given the observed data X and the parameters of the model, we use the forward-backward algorithm to compute the smoothed probabilities of the hidden states C, and then we compute the posterior probabilities of Z given X and C. 
    We also return the most likely sequence of C and Z for each time point. 
    """
    Gamma = create_transition_matrix(gamma_param, beta_param)
    pi_init = np.array([0.0, 0.0, 1.0], dtype=float)

    B = emission_matrix(X, alpha_param, lambda0, lambda1)

    alpha_filtered, scale = forward_C(B, Gamma, pi_init)
    beta_msg = backward_C(B, scale, Gamma)
    qC = compute_smoothed_prob(alpha_filtered, beta_msg)
    qZ = posterior_Z(X, qC, alpha_param, lambda0, lambda1)

    return {
        "Gamma": Gamma,
        "pi_init": pi_init,
        "B": B,
        "alpha_filt": alpha_filtered,
        "beta_msg": beta_msg,
        "qC": qC,
        "qZ": qZ
    }