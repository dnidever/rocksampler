import os
import numpy as np

# Sampler for SSO orbits of tracket data

def model(xdata,pars):
    """
    Generate model for xdata and pars

    Parameters
    ----------
    xdata : numpy array
       Times array in JD (days).
    pars : numpy array
       Parameters for the SSO orbit.
          q=2.011686358249685E+00, 
          e=3.358906223441304E+00, 
          inc=4.406249095001966E+01, 
          node=3.080990336326165E+02, 
          arg=2.091658851414444E+02, 
          M=3.947846842756840E+02, 

    Returns
    -------
    ra : numpy array
       Right ascension.
    dec : numpy array
       Declination.

    Examples
    --------

    ra,dec = model(times,pars)

    """

def sample(data):
    """
    Sample orbits for tracklet data
    """

    # Perform rejection sampling

    # Are there any linear terms that we can optimize over?
    
    pass
