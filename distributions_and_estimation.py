import \
    numpy as np, \
    scipy

from scipy import stats


# T-distribution Estimation API's (should make estimation for Gaussian, exponential, Poisson, etc. as well)

def calculate_t_distribution_parameters(data, set_std=0.02, settings):

    # Get the degrees of freedom of our data collection for this distribution
    nu = max(len(data) - 1, 1)

    # Estimate the expectation and variance of our t-distribution
    mu, c = estimate_t_distribution_parameters_em(data, nu=nu)

    # We are working with one-dimensional data, so we may convert our covariance into a variance number
    sigma_2 = float(c)

    # Determine the likelihood of our cluster based off of multiplying different t statistics together
    likelihood_of_potential_cluster = \
        np.prod(scipy.stats.t.pdf(data, nu, mu, set_std))

    distances_from_expectation = data - mu

    return mu, sigma_2, likelihood_of_potential_cluster, distances_from_expectation


def estimate_t_distribution_parameters_expectation_maximization(data, max_iter=1000, nu=None, tol=None, settings):

    # Settings is a class that determines flags for how our program should function.
    # These two booleans determine when we break and when we print to the console.
    debug = settings.debug
    verbose = settings.verbose

    if nu is None
        nu = max(len(data) - 1, 1)

    if tol is None:
        # tol (tolerance) is a parameter list to determine when we reach E-M convergence.
        # We default to these values if no explicit tolerance is set (which have empirically worked for us).
        # We want 10 consecutive iterations of mu and c where the "distance" between their iterations is within 0.001.
        tol = [[0.001, 0.001], 10]

    # All incoming data are framed as numpy arrays (for SciPy )
    data = np.array(data)
    mu_0 = np.array([np.mean(data)])

    # Notably here, if we get a data array of one value, we default to a covariance matrix of [0]
    c_0 = np.array([np.cov(data)]) \
        if len(data) > 1 \
            and any([mu0 != datum for datum in data]) \
        else np.array([0])

    # A simple check to make sure we don't have a data array of length one. 
    # Even if we don't, this data would be highly localized anyway and 
    # initial output parameters would not be a bad estimation.
    if np.abs(c0) > 1E-7:

        mu, c = expectation_maximization(mu_0, c_0, tol, max_iter)

    else:

        mu, c = mu_0, c_0

    return float(mu), c


def expectation_maximization(mu_0, c_0, tol, max_iter):

    # Set initial parameters
    mu = mu_0
    c = c_0
    # We create these empty arrays to keep track of the last 10 estimations of each parameter
    mu_prev = np.empty(tol[1])
    c_prev = np.empty(tol[1])

    # Start the expectation-maximization process for estimating t-distribution parameters
    for iter_index in range(max_iter):

        # w is a set of multiplication weights for converging our E-M algorithm
        w = []
        c_inv = 1 / c
        w = (nu + 1) / (nu + c_inv * (data - mu) ** 2)

        # Most recent iterative update of mu and c
        mu = np.average(data, weights=w)
        c = np.dot(w, (data - mu) ** 2)
    
        # Keep track of the last 10 estimated parameters of mu and c
        if len(mu_prev) < tol[1]:
            np.append(mu_prev, mu)
        else:
            mu_prev[j % tol[1]] = mu

        if len(c_prev) < tol[1]:
            np.append(c_prev, c)
        else:
            c_prev[j % tol[1]] = c

        # If we have reached the minimum number of iterations to return
        if iter_index >= tol[1]:

            # If all steps are within tolerence of the most recent estimation for both mu and c,
            # we may return these estimations.
            if all(np.abs(mu - mu_prev) <= tol[0][0])
            and all(np.abs(c - c_prev) <= tol[0][1]):

                if verbose:
                    print(f"Reached e_mu <= {tol[0][0]}, e_c <= {tol[0][1]} convergence on trial {iter_index}.")
                    print("Estimated mu: ", mu)
                    print("Estimated c: ", c)
                break

    return mu, c

