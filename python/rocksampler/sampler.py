import os
import numpy as np

from spacerocks import SpaceRock, Units

from astroquery.jplhorizons import Horizons

from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord

units = Units()
units.timescale = 'tdb' # JPL gives elements in TDB. This actually matters for precise ephemerides.
#units.current()

# SSO model

class SSO(object):

    def __init__(self,e=None,a=None,inc=None,node=None,
                 arg=None,M=None,epoch=None,origin='ssb',obscode='W84'):
        self.e = e
        self.a = a
        self.inc = inc
        self.node = node
        self.arg = arg
        self.M = M
        self.epoch = epoch
        self.origin = origin
        self.obscode = obscode
        for c in ['e','a','inc','mode','arg','M','epoch']:
            if eval(c) is None:
                raise ValueError('Need value for '+c)

        # Check the values
        assert 0<=self.e<=1
        assert 0<self.a
        assert 0<=self.inc<=180
        assert 0<=self.node<=360
        assert 0<=self.arg<=360        
        assert 0<=self.M<=360
        assert 0<self.epoch
        
    @property
    def q(self):
        """ Return the periapsis."""
        return self.a*(1-self.e)
            
    def __repr__(self):
        out = '<SSO ['
        vals = []
        for c in ['e','a','inc','mode','arg','M','epoch']:
            vals.append('{:}={:.3f}'.format(c,getattr(self,c)))
        out += ','.join(vals)
        out += ']>'
        return out
            
    def __call__(self,times,obscode='W84'):
        # Initialize the spacerock object
        rock = SpaceRock(q=self.q,e=self.e,
                         inc=self.inc,node=self.node,
                         arg=self.arg,M=self.M,
                         epoch=self.epoch,
                         name='test',
                         origin=self.origin,
                         units=units)
        units.timescale = 'utc'
        prop, planets, sim = rock.propagate(epochs=times, model='PLANETS', units=units)        
        obs = prop.observe(obscode=obscode)
        return obs.ra.deg,obs.dec.deg

# Sampler for SSO orbits of tracket data

class Sampler(object):

    def __init__(self,t=None,ra=None,dec=None,ra_error=None,dec_error=None,obscode='W84'):
        self.t = t                    # JD time in days
        self.ra = ra                  # ra in degrees
        self.dec = dec                # dec in degrees
        self.ra_error = ra_error      # in arcsec
        self.dec_error = dec_error    # in arcsec
        self.obscode = obscode

    def model(self,pars):
        """
        Generate model for pars
        
        Parameters
        ----------
        pars : numpy array
           Parameters for the SSO orbit.
             [e,a,inc,node,arg,M]

        Returns
        -------
        ra : numpy array
           Right ascension.
        dec : numpy array
           Declination.
        
        Examples
        --------
        
        ra,dec = model(pars)
        
        """

        if self.t is None:
            raise ValueError('No time values found.  Add them to the sampler.')

        if len(pars)<6:
            raise ValueError('pars must have 6 or 7 elements')

        if len(pars)==6:
            e,a,inc,node,arg,M = pars
        elif len(pars)==7:
            e,a,inc,node,arg,M,epoch = pars
        
        # times must be JD in days
        times = self.t
        if len(pars)==7:
            epoch = pars[6]
        else:
            epoch = times[0]

        # Calculate perapsis
        q = a*(1-e)
            
        # Initialize the spacerock object
        rock = SpaceRock(q=q,e=e,inc=inc,node=node
                         arg=arg,M=M,epoch=epoch,
                         name='test',origin='ssb',
                         units=units)

        # Propagate to the times
        units.timescale = 'utc'
        prop, planets, sim = rock.propagate(epochs=times, model='PLANETS', units=units)        
        obs = prop.observe(obscode=self.obscode)
        
        #rock2 = rock.analytic_propagate(epoch=xdata[0], units=units)
        #eph = rock.observe(obscode=self.obscode)
        #obs = rock.xyz_to_tel(obscode)
        #eph = Ephemerides(x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, epoch=self.epoch, name=self.name)
        #pos_pred = SkyCoord(obs.ra.deg, obs.dec.deg, frame='icrs', unit=(u.deg, u.deg))

        return obs.ra.deg,obs.dec.deg

    def likelihood(self,pars):
        """
        Calculate the log likelihood of the data given the model
        """

        mra,mdec = self.model(xdata,pars)
        pos_obs = SkyCoord(self.ra, self.dec, frame='icrs', unit=(u.deg, u.deg))
        pos_model = SkyCoord(mra, mdec, frame='icrs', unit=(u.deg, u.deg))
        sep = pos_obs.separation(pos_model).arcsec
        sigma2 = ra_error**2 + dec_error**2  # in arcsec
        return -0.5 * np.sum(sep ** 2 / sigma2 + np.log(sigma2))
        
    #def log_likelihood(theta, x, y, yerr):
    #    m, b, log_f = theta
    #    model = m * x + b
    #    sigma2 = yerr**2 + model**2 * np.exp(2 * log_f)
    #    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))
        
        
def sample(data):
    """
    Sample orbits for tracklet data
    """

    # Perform rejection sampling

    # Are there any linear terms that we can optimize over?

    startdate = Time('2021-01-01', scale='utc', format='iso')
    testdates = Time(np.arange(startdate.jd, startdate.jd + 5 * 365.25, 30), scale='utc', format='jd')

    # Parameters to sample:
    # e (eccentricity): [0,1]    
    # a (semi-major axis): [0,+inf]
    # inc (inclination): [0,180]
    # node (longitude of the ascending node): [0,360?]
    # arg (argument of periapsis): [0,360?]
    # M (mean anomaly at epoch): [0,360]
    # epoch (reference epoch): [-inf,+inf]
