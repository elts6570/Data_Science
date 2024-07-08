import arviz as az
import bambi as bmb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm


def read_data():
    """
    This function reads data from # http://www.physics.uwyo.edu/~jessicas/teach/astron/cephiedperiodlum.pdf.
    We expect a ~ -3, b ~ -4 such that M = a[log10(P)-1] + b

    Returns
    -------
    data : pandas dataset
        period (x) vs absolute magnitude (y) data 
    """
    df = pd.read_csv('Cepheids.csv', delimiter=' ')
    data = pd.DataFrame({'x': df['P']})
    data['y'] = df['M']
    data['log10_x_minus_1'] = np.log10(data['x']) - 1
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
    "Intercept": bmb.Prior("Uniform", lower = -10, upper = 10),    
    "log10_x_minus_1": bmb.Prior("Uniform", lower = -10, upper = 10) 
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
    model = bmb.Model("y ~ log10_x_minus_1", data, priors = priors)

    trace = model.fit(draws=1000, tune=1000, chains=4, progressbar=True)
    return trace


def diagnostics(trace):
    """
    This function plots the chains' summary statistics (mean, sd, hdi etc.) and traceplots for each
    parameter.
    """
    print("Summary statistics:")
    summary = az.summary(trace)
    print(summary)

    print("Trace plots:")
    az.plot_trace(trace)
    plt.show() 
    return None


if __name__ == "__main__":
    data = read_data()
    priors = set_priors()
    trace = sampler(priors, data)
    diagnostics(trace)
