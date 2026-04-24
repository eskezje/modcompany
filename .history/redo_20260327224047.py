import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans

def create_transition_matrix(gamma_param: float, beta_param: float) -> np.ndarray:
    Gamma = np.array([
            [(1-gamma_param)  ,0          ,gamma_param],
            [0          ,(1-gamma_param)  ,gamma_param],
            [(beta_param/2)   ,(beta_param/2)   ,(1-beta_param)]
            ], dtype=float)
    return Gamma

def x_given_z_vector(x_val: int, lambda0: float, lambda1: float) -> np.ndarray:
    return np.array([
        poisson.pmf(x_val, mu=lambda0),  # z=0
        poisson.pmf(x_val, mu=lambda1)   # z=1
    ], dtype=float)

def z_given_c_table(alpha_param: float) -> np.ndarray:
    pi = np.array([1 -alpha_param, alpha_param, 0.5], dtype=float)  # P(Z=1|C=c)

    phi_zc = np.zeros((2, 3), dtype=float)
    phi_zc[1, :] = pi
    phi_zc[0, :] = 1.0 - pi
    return phi_zc

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


def simulate_z(C: np.ndarray, n: int, alpha_param: float, rng=None) -> np.ndarray:
    """
    Simulates the Z states for T times, for n neurons, given the C states and the alpha parameter.
    """
    rng = np.random.default_rng() if rng is None else rng
    T = len(C)
    Z = np.empty((T, n), dtype=int)

    phi_zc = z_given_c_table(alpha_param)

    for t in range(T):
        for i in range(n):
            p_z1_given_c = phi_zc[1, C[t]]
            Z[t, i] = rng.binomial(1, p_z1_given_c)

    return Z

def simulate_x(Z: np.ndarray, lambda0: float, lambda1: float, rng=None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng 
    lamda = lambda0 + (lambda1 - lambda0) * Z
    X = rng.poisson(lam=lamda)
    return X

def simulate_hmm(T: int, n: int, alpha_param: float, beta_param: float, gamma_param: float, lambda_0:float , lambda_1:float, seed=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

def make_datasets(M: int, t_index: int, T: int, n: int, alpha_param: float, beta_param: float, gamma_param :float, lambda_0: float, lambda_1: float, seed=123) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulates M independent HMM sequences, where we save the features (spike counts) and labels (hidden states) at specific time indexes, as well as the full sequences of C, Z and X for all M experiments.
    """
    rng = np.random.default_rng(seed)

    x_features = np.empty((M, n), dtype=int)      # one row per experiment 
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

def message_z_to_c(x_val: int, alpha_param: float, lambda0: float, lambda1: float) -> np.ndarray:
    phi_zc = z_given_c_table(alpha_param)
    phi_xz = x_given_z_vector(x_val, lambda0, lambda1)
    # sum over z
    msg = np.sum(phi_zc * phi_xz[:, None], axis=0)
    return msg


def local_evidence_messages(X: np.ndarray, alpha_param: float, lambda0: float, lambda1: float) -> tuple[np.ndarray, np.ndarray]:
    T, n = X.shape
    m_z_to_c = np.zeros((T, n, 3), dtype=float)

    for t in range(T):
        for i in range(n):
            m_z_to_c[t, i] = message_z_to_c(
                x_val=int(X[t, i]),
                alpha_param=alpha_param,
                lambda0=lambda0,
                lambda1=lambda1
            )

    psi = np.prod(m_z_to_c, axis=1)  # multiply over neurons i
    return m_z_to_c, psi


def forward_messages(psi: np.ndarray, Gamma: np.ndarray, pi_init: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T = psi.shape[0]
    alpha = np.zeros((T, 3), dtype=float)
    scale = np.zeros(T, dtype=float)

    alpha[0] = pi_init * psi[0]
    scale[0] = alpha[0].sum()
    alpha[0] /= scale[0]

    for t in range(1, T):
        alpha[t] = psi[t] * (alpha[t - 1] @ Gamma)
        scale[t] = alpha[t].sum()
        alpha[t] /= scale[t]

    return alpha, scale

def backward_messages(psi: np.ndarray, Gamma: np.ndarray, scale: np.ndarray) -> np.ndarray:
    T = psi.shape[0]
    beta = np.zeros((T, 3), dtype=float)
    beta[-1] = 1.0

    for t in range(T - 2, -1, -1):
        beta[t] = Gamma @ (psi[t + 1] * beta[t + 1])
        beta[t] /= scale[t + 1]

    return beta

def posterior_c(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    qC = alpha * beta
    qC /= qC.sum(axis=1, keepdims=True)
    return qC

def posterior_z(X: np.ndarray, qC: np.ndarray, alpha_param: float, lambda0: float, lambda1: float) -> np.ndarray:
    T, n = X.shape
    phi_zc = z_given_c_table(alpha_param)

    qZ = np.zeros((T, n), dtype=float)

    for t in range(T):
        for i in range(n):
            phi_xz = x_given_z_vector(int(X[t, i]), lambda0, lambda1)
            local = phi_zc * phi_xz[:, None] #proportional to P(Z=z, x | C=c)

            denom = local.sum(axis=0)
            p_z1_given_xc = local[1, :] / denom

            qZ[t, i] = np.sum(p_z1_given_xc * qC[t])

    return qZ

def hmm_pipeline(X: np.ndarray, alpha_param: float, beta_param: float, gamma_param: float, lambda0: float, lambda1: float) -> dict[str, np.ndarray]:
    Gamma = create_transition_matrix(gamma_param, beta_param)
    pi_init = np.array([0.0, 0.0, 1.0], dtype=float)

    m_z_to_c, psi = local_evidence_messages(X, alpha_param, lambda0, lambda1)
    alpha_msg, scale = forward_messages(psi, Gamma, pi_init)
    beta_msg = backward_messages(psi, Gamma, scale)
    qC = posterior_c(alpha_msg, beta_msg)
    qZ = posterior_z(X, qC, alpha_param, lambda0, lambda1)

    c_hat = np.argmax(qC, axis=1)
    z_hat = (qZ > 0.5).astype(int)

    return {
        "Gamma": Gamma,
        "pi_init": pi_init,
        "m_z_to_c": m_z_to_c,   # explicit local messages from Z to C
        "psi": psi,             # local evidence at each time t
        "alpha_msg": alpha_msg,
        "beta_msg": beta_msg,
        "qC": qC,
        "qZ": qZ,
        "c_hat": c_hat,
        "z_hat": z_hat,
        "scale": scale
    }

def plot_heatmap_z(z_true: np.ndarray, qZ: np.ndarray) -> None:
    """
    Plots a heatmap of the true and estimated Z values over time and neurons.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    im0 = axes[0].imshow(z_true.T, aspect="auto", cmap="plasma")
    axes[0].set_title("True Z")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Neuron")
    fig.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(qZ.T, aspect="auto", cmap="plasma")
    axes[1].set_title("Estimated Z")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Neuron")
    fig.colorbar(im1, ax=axes[1])
    
    plt.tight_layout()
    plt.show()

def plot_c_posterior(result) -> None:
    plt.figure(figsize=(12,4))
    plt.plot(result["qC"][:,0], label=r"$P(C_t=0\mid X)$")
    plt.plot(result["qC"][:,1], label=r"$P(C_t=1\mid X)$")
    plt.plot(result["qC"][:,2], label=r"$P(C_t=2\mid X)$")
    plt.xlabel("t")
    plt.ylabel("posterior probability")
    plt.title("Posterior probabilities for $C_t$ on simulated data")
    plt.legend()
    plt.grid(alpha=0.6)
    plt.show()


def plot_heatmap_z_posterior(result) -> None:
    plt.figure(figsize=(12,4))
    plt.imshow(result["qZ"].T, aspect="auto", cmap="plasma")
    plt.colorbar(label=r"$P(Z_{t,i}=1\mid X)$")
    plt.xlabel("Time")
    plt.ylabel("Neuron")
    plt.title("Posterior heatmap for $Z_{t,i}$ on simulated data")
    plt.show()

def lambda_hat_from_xz(X: np.ndarray, Z_true: np.ndarray) -> tuple[float, float]:
    """
    A simple function to learn the lambda parameters for the emission distribution, given the observed data X and the true Z values.
    we will take lambda0 as the average of Z_t,i=0 and lambda1 as the average of Z_t,i=1, using the true Z values.
    """
    lambda0 = 0.0
    lambda1 = 0.0
    counts0 = 0
    counts1 = 0
    for t in range(X.shape[0]):
        for i in range(X.shape[1]):
            if Z_true[t, i] == 0:
                lambda0 += X[t, i]
                counts0 += 1
            else:
                lambda1 += X[t, i]
                counts1 += 1
    lambda0 = lambda0 / counts0 if counts0 > 0 else 0.0
    lambda1 = lambda1 / counts1 if counts1 > 0 else 0.0
    return lambda0, lambda1

def alpha_hat_from_cz(C: np.ndarray, Z: np.ndarray) -> float:
    """
    Gets the average value of Z_t,i=1 when C_t=1, and the average value of Z_t,i=0 when C_t=0, to learn the alpha parameter for the emission distribution, given the true C and Z values.
    """
    alpha_num = 0 
    alpha_den = 0 

    for t in range(len(C)):
        if C[t] == 0:
            alpha_num += np.sum(Z[t] == 0)
            alpha_den += Z.shape[1]
        elif C[t] == 1:
            alpha_num += np.sum(Z[t] == 1)
            alpha_den += Z.shape[1]

    return alpha_num/alpha_den if alpha_den > 0 else 0.0

def beta_hat_from_C(C: np.ndarray) -> float:
    """
    Gets the average value of C_t+1=1 or 0 when C_t=2, to learn the beta parameter for the transition distribution, given the true C values.
    """
    seen = 0
    from_2 = 0
    for t in range(len(C)-1):
        if C[t] == 2:
            from_2 += 1 
            if C[t+1] == 1 or C[t+1] == 0:
                seen +=1 

    return seen/from_2 if from_2 > 0 else 0.0

def gamma_hat_from_C(C: np.ndarray) -> float:
    """
    Gets the average value of C_t+1=2 when C_t=0 or 1, to learn the gamma parameter for the transition distribution, given the true C values.
    """
    from_0_1 = 0
    seen = 0
    for t in range(len(C)-1):
        if C[t] == 0 or C[t] == 1:
            from_0_1 += 1
            if C[t+1] == 2:
                seen += 1
    return seen/from_0_1 if from_0_1 > 0 else 0.0


def learn_all_params_from_known_data(X: np.ndarray, C: np.ndarray, Z: np.ndarray) -> dict[str, float]:
    """
    Tries to lean all the parameters of the model given the complete data, meaning the observed X and the true hidden states C and Z. 
    """
    lambda0_hat, lambda1_hat = lambda_hat_from_xz(X, Z)
    alpha_hat = alpha_hat_from_cz(C, Z)
    beta_hat = beta_hat_from_C(C)
    gamma_hat = gamma_hat_from_C(C)

    return {
        "alpha_hat": alpha_hat,
        "beta_hat": beta_hat,
        "gamma_hat": gamma_hat,
        "lambda0_hat": lambda0_hat,
        "lambda1_hat": lambda1_hat
    }

def hard_assigment_EM(
                    X: np.ndarray, 
                    alpha_param_init: float | None = None,
                    beta_param_init: float | None = None, 
                    gamma_param_init: float | None = None, 
                    lambda0_init: float | None = None, 
                    lambda1_init: float | None = None,
                    max_iter: int = 500,
                    change_threshold: float = 1e-7):

    if lambda0_init is None or lambda1_init is None:
        lambda0_init, lambda1_init, n0, n1 = init_lambda_kmeans(X)
    else:
        n0 = None 
        n1 = None 
    
    if alpha_param_init is None:
        alpha_param_init = 0.9

    if beta_param_init is None:
        beta_param_init = 0.1

    if gamma_param_init is None:
        gamma_param_init = 0.1

    alpha_cur = alpha_param_init
    beta_cur = beta_param_init
    gamma_cur = gamma_param_init
    lambda0_cur = lambda0_init
    lambda1_cur = lambda1_init

    previous_runs = []

    for i in range(max_iter):
        res = hmm_pipeline(X, alpha_cur, beta_cur, gamma_cur, lambda0_cur, lambda1_cur)
        c_hat = res["c_hat"]
        z_hat = res["z_hat"]

        new_params = learn_all_params_from_known_data(X, c_hat, z_hat)

        alpha_new = new_params["alpha_hat"]
        beta_new = new_params["beta_hat"]
        gamma_new = new_params["gamma_hat"]
        lambda0_new = new_params["lambda0_hat"]
        lambda1_new = new_params["lambda1_hat"]

        delta = max(abs(alpha_new - alpha_cur), abs(beta_new - beta_cur), abs(gamma_new - gamma_cur), abs(lambda0_new - lambda0_cur), abs(lambda1_new - lambda1_cur))
        if delta < change_threshold:
            print(f"We wont change much more, and have converged at iteration {i}.")
            break
        previous_runs.append({
            "iteration": i,
            "alpha": alpha_new,
            "beta": beta_new,
            "gamma": gamma_new,
            "lambda0": lambda0_new,
            "lambda1": lambda1_new
        })

        alpha_cur = alpha_new
        beta_cur = beta_new
        gamma_cur = gamma_new
        lambda0_cur = lambda0_new
        lambda1_cur = lambda1_new

    final_result = hmm_pipeline(X, alpha_cur, beta_cur, gamma_cur, lambda0_cur, lambda1_cur)

    return {
        "alpha_hat": alpha_cur,
        "beta_hat": beta_cur,
        "gamma_hat": gamma_cur,
        "lambda0_hat": lambda0_cur,
        "lambda1_hat": lambda1_cur,
        "previous_runs": previous_runs,
        "final_res": final_result
    }


def init_lambda_kmeans(X: np.ndarray) -> tuple[float, float, int, int]:
    x_vals = X.reshape(-1, 1)

    kmeans = KMeans(n_clusters=2, random_state=0)
    estimated_labels = kmeans.fit_predict(x_vals)

    cluster0 = []
    cluster1 = []

    for i in range(len(estimated_labels)):
        if estimated_labels[i] == 0:
            cluster0.append(x_vals[i, 0])
        else:
            cluster1.append(x_vals[i, 0])

    mean0 = np.mean(cluster0)
    mean1 = np.mean(cluster1)

    if mean0 < mean1:
        lambda0_guess = mean0
        lambda1_guess = mean1
        n0 = len(cluster0)
        n1 = len(cluster1)
    else:
        lambda0_guess = mean1
        lambda1_guess = mean0
        n0 = len(cluster1)
        n1 = len(cluster0)

    return float(lambda0_guess), float(lambda1_guess), n0, n1


def plot_convergence(em_res):
    runs = em_res["previous_runs"]

    alphas = [run["alpha"] for run in runs]
    betas= [run["beta"] for run in runs]
    gammas = [run["gamma"] for run in runs]
    lambda0s = [run["lambda0"] for run in runs]
    lambda1s = [run["lambda1"] for run in runs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(alphas, label=r"$\alpha$", zorder=2)
    ax1.plot(betas, label=r"$\beta$", zorder=2)
    ax1.plot(gammas, label=r"$\gamma$", zorder=2)
    ax1.set_xlabel("EM iteration")
    ax1.set_ylabel("Probability")
    ax1.grid(alpha=0.3, zorder=1)
    ax1.legend()

    ax2.plot(lambda0s, label=r"$\lambda_0$", zorder=2)
    ax2.plot(lambda1s, label=r"$\lambda_1$", zorder=2)
    ax2.set_xlabel("EM iteration")
    ax2.set_ylabel("Lambda value")
    ax2.grid(alpha=0.3, zorder=1)
    ax2.legend()

    plt.tight_layout()
    plt.show()