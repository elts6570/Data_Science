import arviz as az
import bambi as bmb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm


def simulate_data():
    """
    This function generates synthetic data with a given model.

    Returns
    -------
    data : pandas dataset
        date data 
    """
    np.random.seed(42)
    x = np.linspace(0.001, 10, 1000)

    data = pd.DataFrame({"x": x})

    beta_0 = 0.4
    beta_1 = 0.2
    beta_2 = -1.5
   
    data["y"] = beta_0 + beta_1 * data["x"] + beta_2 * np.log10(data["x"]) + np.random.normal(0, 0.01, len(x))
    return data


def set_priors():
    """
    This function sets the priors on each free parameter.

    Returns
    -------
    priors : dict
        bambi priors on each free parameter
    """
    priors = {
    "Intercept": bmb.Prior("Uniform", lower = 0, upper = 10),  # Prior for beta_0
    "x": bmb.Prior("Uniform", lower = 0, upper=10),          # Prior for beta_1
    "np.log10(x)": bmb.Prior("Uniform", lower = -10, upper = 10) # Prior for beta_2
    }
    return priors


def sampler(priors, data):
    """
    This function creates the bambi model and draws samples using a NUTS sampler.

    Returns
    -------
    trace : arciz inference_data 
        results of the model sampling containing all MCMC chains
    """
    model = bmb.Model("y ~ x + np.log10(x)", data, priors = priors)

    trace = model.fit(draws=1000, tune=1000, chains=4, progressbar=True)
    return trace


def diagnostics(trace):
    """
    This function plots the chains' summary statistics (mean, sd, hdi etc.), traceplots for each
    parameter, autocorrelation functions, Gelman-Rubin diagnostics and an estimate of the effective
    sample size.
    """
    print("Summary statistics:")
    summary = az.summary(trace)
    print(summary)
    
    print("Trace plots:")
    az.plot_trace(trace)
    plt.show()
    
    print("Autocorrelation plots:")
    az.plot_autocorr(trace)
    plt.show()
    
    print("Gelman-Rubin diagnostic (R-hat):")
    rhat = az.rhat(trace)
    print(rhat)
    
    print("Effective sample size (ESS):")
    ess = az.ess(trace)
    print(ess)
    
    return None


if __name__ == "__main__":
    data = simulate_data()
    priors = set_priors()
    trace = sampler(priors, data)
    diagnostics(trace)
