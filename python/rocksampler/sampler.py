import os
import numpy as np
from dlnpyutils import utils as dln,coords

from spacerocks import SpaceRock, Units

from astroquery.jplhorizons import Horizons

from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord

units = Units()
units.timescale = 'tdb' # JPL gives elements in TDB. This actually matters for precise ephemerides.
#units.current()

# SSO model

def proper_motion(t,ra,dec,ra_error=0.1,dec_error=0.1):
    raerr = np.array(ra_error,np.float64)    # arcsec
    ra = np.array(ra,np.float64)
    ra -= np.mean(ra)
    ra *= 3600 * np.cos(np.deg2rad(np.mean(dec)))     # convert to true angle, arcsec
    t = t.copy()
    t -= np.mean(t)
    t *= 24   # in hours
    #t /= 365.2425                          # convert to year
    # Calculate robust slope
    pmra, pmraerr = dln.robust_slope(t,ra,raerr,reweight=True)
    # in arcsec/hour
    
    decerr = np.array(dec_error,np.float64)   # arcsec
    dec = np.array(dec,np.float64)
    dec -= np.mean(dec)
    dec *= 3600                         # convert to arcsec
    # Calculate robust slope
    pmdec, pmdecerr = dln.robust_slope(t,dec,decerr,reweight=True)
    # in arcsec/hour
    return pmra,pmraerr,pmdec,pmdecerr
    
    
class SSO(object):
    """
    Solar system orbject

    origin : str, optional
       Origin of the reference frame: 'ssb'
    frame : str, optional
       Type of reference frame coordinate system:
         'eclipJ2000'
    """
    
    
    def __init__(self,e=None,a=None,inc=None,node=None,
                 arg=None,M=None,epoch=None,origin='ssb',
                 frame='eclipJ2000',obscode='W84',analytic=True):
        self.e = e
        self.a = a
        self.inc = inc
        self.node = node
        self.arg = arg
        self.M = M
        self.epoch = epoch
        self.origin = origin
        self.frame = frame
        self.obscode = obscode
        self.analytic = analytic
        for c in ['e','a','inc','node','arg','M','epoch']:
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
        for c in ['e','a','inc','node','arg','M','epoch']:
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
                         frame=self.frame,
                         units=units)
        units.timescale = 'utc'
        if self.analytic:
            rock = rock.analytic_propagate(epoch=times, units=units)
            obs = rock.observe(obscode=self.obscode)
        else:
            prop, planets, sim = rock.propagate(epochs=times, model='PLANETS', units=units)        
            obs = prop.observe(obscode=obscode)
        return obs #.ra.deg,obs.dec.deg

# Sampler for SSO orbit tracklets
class TrackletSampler(object):

    def __init__(self,t=None,ra=None,dec=None,ra_error=None,dec_error=None,
                 obscode='W84',analytic=False):
        self.t = t                    # JD time in days
        self.ra = ra                  # ra in degrees
        self.dec = dec                # dec in degrees
        self.ra_error = ra_error      # in arcsec
        self.dec_error = dec_error    # in arcsec
        self.obscode = obscode
        self.analytic = analytic

        self._mnt = None        
        self._mnra = None
        self._mnra_error = None
        self._mndec = None
        self._mndec_error = None
        self._pmra = None
        self._pmra_error = None
        self._pmdec = None
        self._pmdec_error = None        

    def __call__(self,pars):
        """
        Generate model for pars
        
        Parameters
        ----------
        pars : numpy array
           Parameters for the SSO orbit.
             [e,a,inc,node,arg,M]

        Returns
        -------
        eph : Ephemeris
           Ephemeris information.
        
        Examples
        --------
        
        eph = model(pars)
        
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
            epoch = self.mnt

        # Calculate perapsis
        q = a*(1-e)
            
        # Initialize the spacerock object
        rock = SpaceRock(q=q,e=e,inc=inc,node=node,
                         arg=arg,M=M,epoch=epoch,
                         name='test',origin='ssb',
                         units=units)
        # Propagate to the times
        units.timescale = 'utc'
        if self.analytic:
            rock = rock.analytic_propagate(epoch=self.mnt, units=units)
            eph = rock.observe(obscode=self.obscode)
        else:
            prop, planets, sim = rock.propagate(epochs=self.mnt, model='PLANETS', units=units)
            eph = prop.observe(obscode=self.obscode)
        return eph
            
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
        ra : float
           Right ascension in degrees.
        dec : float
           Declination in degrees.
        pmra : float
           Proper motion in RA in arcsec/hour.
        pmdec : float
           Proper motion in DEC in arcsec/hour.
        
        Examples
        --------
        
        ra,dec,pmra,pmdec = model(pars)
        
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
            epoch = self.mnt

        # Calculate perapsis
        q = a*(1-e)
            
        # Initialize the spacerock object
        rock = SpaceRock(q=q,e=e,inc=inc,node=node,
                         arg=arg,M=M,epoch=epoch,
                         name='test',origin='ssb',
                         units=units)

        # Propagate to the times
        units.timescale = 'utc'
        if self.analytic:
            rock = rock.analytic_propagate(epoch=self.mnt, units=units)
            #rock = rock.analytic_propagate(epoch=times, units=units)            
            eph = rock.observe(obscode=self.obscode)
        else:
            prop, planets, sim = rock.propagate(epochs=self.mnt, model='PLANETS', units=units)
            #prop, planets, sim = rock.propagate(epochs=times, model='PLANETS', units=units)            
            eph = prop.observe(obscode=self.obscode)
        
        #rock2 = rock.analytic_propagate(epoch=xdata[0], units=units)
        #eph = rock.observe(obscode=self.obscode)
        #obs = rock.xyz_to_tel(obscode)
        #eph = Ephemerides(x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, epoch=self.epoch, name=self.name)
        #pos_pred = SkyCoord(obs.ra.deg, obs.dec.deg, frame='icrs', unit=(u.deg, u.deg))

        out = (eph.ra.deg,eph.dec.deg,eph.ra_rate.to(u.arcsec/u.hour).value,
               eph.dec_rate.to(u.arcsec/u.hour).value)
        return out

    @property
    def mnt(self):
        """ Calculate and return the mean time of this tracklet."""
        if self._mnt is None:
            if self.t is None:
                raise ValueError('No T values')
            self._mnt = np.mean(self.t)
        return self._mnt

    @property
    def mnra(self):
        """ Calculate and return the mean ra of this tracklet."""
        if self._mnra is None:
            if self.ra is None:
                raise ValueError('No RA values')
            self._mnra = np.mean(self.ra)
        return self._mnra

    @property
    def mnra_error(self):
        """ Calculate and return the mean dec error of this tracklet."""
        if self._mnra_error is None:
            if self.ra_error is None:
                raise ValueError('No RA_ERROR values')
            self._mnra_error = np.sqrt(np.sum(self.ra_error**2))
        return self._mnra_error

    @property
    def mndec(self):
        """ Calculate and return the mean dec of this tracklet."""
        if self._mndec is None:
            if self.dec is None:
                raise ValueError('No DEC values')
            self._mndec = np.mean(self.dec)
        return self._mndec

    @property
    def mndec_error(self):
        """ Calculate and return the mean dec error of this tracklet."""
        if self._mndec_error is None:
            if self.dec_error is None:
                raise ValueError('No DEC_ERROR values')
            self._mndec_error = np.sqrt(np.sum(self.dec_error**2))
        return self._mndec_error

    def _proper_motion(self):
        """ Calculate the proper motions."""
        if (self.t is None or self.ra is None or self.ra_error is None or
            self.dec is None or self.dec_error is None):
                raise ValueError('No T, RA, RA_ERROR, DEC or DEC_ERROR values')
        pmra,pmraerr,pmdec,pmdecerr = proper_motion(self.t.copy(),self.ra.copy(),
                                                    self.dec.copy(),self.ra_error.copy(),
                                                    self.dec_error.copy())
        self._pmra = pmra
        self._pmra_error = pmraerr
        self._pmdec = pmdec
        self._pmdec_error = pmdecerr
        
    @property
    def pmra(self):
        """ Calculate and return the mean pmra of this tracklet."""
        if self._pmra is None:
            self._proper_motion()  # calculate the proper motion data
        return self._pmra

    @property
    def pmra_error(self):
        """ Calculate and return the mean pmra error of this tracklet."""
        if self._pmra_error is None:
            self._proper_motion()  # calculate the proper motion data            
        return self._pmra_error
    
    @property
    def pmdec(self):
        """ Calculate and return the mean pmdec of this tracklet."""
        if self._pmdec is None: 
            self._proper_motion()  # calculate the proper motion data           
        return self._pmdec

    @property
    def pmdec_error(self):
        """ Calculate and return the mean pmdec error of this tracklet."""
        if self._pmdec_error is None:
            self._proper_motion()  # calculate the proper motion data            
        return self._pmdec_error
        
    def likelihood(self,pars):
        """
        Calculate the log likelihood of the data given the model
        """

        mra,mdec,mpmra,mpmdec = self.model(pars)
        cooresid = coords.sphdist(mra,mdec,self.mnra,self.mndec)*3600
        coosigma = np.sqrt(self.mnra_error**2 + self.mndec_error**2)     # in arcsec
        zcoo = cooresid/coosigma
        pmresid = np.sqrt((self.pmra-mpmra)**2 + (self.pmdec-mpmdec)**2)
        pmsigma = np.sqrt(self.pmra_error**2+self.pmdec_error**2)
        zpm = pmresid/pmsigma
        lnlik = -0.5 * np.sum(zcoo + zpm + np.log(coosigma**2) + np.log(pmsigma**2))        
        #pos_obs = SkyCoord(self.mnra, self.mndec, frame='icrs', unit=(u.deg, u.deg))
        #pos_model = SkyCoord(mra, mdec, frame='icrs', unit=(u.deg, u.deg))
        #sep = pos_eph.separation(pos_model).arcsec
        #sigma2 = ra_error**2 + dec_error**2  # in arcsec
        #return -0.5 * np.sum(sep ** 2 / sigma2 + np.log(sigma2))
        return lnlik
        
    #def log_likelihood(theta, x, y, yerr):
    #    m, b, log_f = theta
    #    model = m * x + b
    #    sigma2 = yerr**2 + model**2 * np.exp(2 * log_f)
    #    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))
        
        
def sample(tab,arange=[0,1000],erange=[0,1],irange=[0,180],noderange=[0,360],
           argrange=[0,360],Mrange=[0,360],nsamples=10000,analytic=True):
    """
    Sample orbits for tracklet data
    [e,a,inc,node,arg,M]
    """

    # Perform rejection sampling

    # Are there any linear terms that we can optimize over?
    # maybe with analytic ones
    # maybe only get the ephemeris at the midpoint of the tracklet
    #   and compare the mid-point coordinates and angular vector
    #   should be faster
    # I wonder if we could use Jax here for autoderivative to speed things up?

    # Get time
    if 'time' in tab.colnames:
        t = tab['time']
    elif 't' in tab.colnames:
        t = tab['t']
    elif 'mjd' in tab.colnames:
        t = tab['mjd']+2400000.5
    elif 'jd' in tab.colnames:
        t = tab['jd']

    # Coordinates
    ra = tab['ra']
    dec = tab['dec']
    if 'raerr' in tab.colnames:
        ra_error = tab['raerr']
    elif 'ra_error' in tab.colnames:
        ra_error = tab['ra_error']
    else:
        ra_error = ra*0+0.1
    if 'decerr' in tab.colnames:
        dec_error = tab['decerr']
    elif 'dec_error' in tab.colnames:
        dec_error = tab['dec_error']
    else:
        dec_error = dec*0+0.1        
        
    #startdate = Time('2021-01-01', scale='utc', format='iso')
    #testdates = Time(np.arange(startdate.jd, startdate.jd + 5 * 365.25, 30), scale='utc', format='jd')

    # Parameters to sample:
    # e (eccentricity): [0,1]    
    # a (semi-major axis): [0,+inf]
    # inc (inclination): [0,180]
    # node (longitude of the ascending node): [0,360?]
    # arg (argument of periapsis): [0,360?]
    # M (mean anomaly at epoch): [0,360]
    # epoch (reference epoch): [-inf,+inf]

    ts = TrackletSampler(t=t,ra=ra,dec=dec,ra_error=ra_error,
                         dec_error=dec_error,analytic=analytic)

    # Create the samples
    esamples = np.random.uniform(erange[0],erange[1],nsamples)
    asamples = np.random.uniform(arange[0],arange[1],nsamples)
    isamples = np.random.uniform(irange[0],irange[1],nsamples)
    nodesamples = np.random.uniform(noderange[0],noderange[1],nsamples)
    argsamples = np.random.uniform(argrange[0],argrange[1],nsamples)
    Msamples = np.random.uniform(Mrange[0],Mrange[1],nsamples)    
    
    # Calculate the likelihood
    dt = [('pars',float,6),('lnlik',float)]
    res = np.zeros(nsamples,dtype=np.dtype(dt))
    for i in range(nsamples):
        pars = [esamples[i],asamples[i],isamples[i],
                nodesamples[i],argsamples[i],Msamples[i]]
        res['pars'][i] = pars
        res['lnlik'][i] = ts.likelihood(pars)
        #print(i,pars,lnlik[i])

        #import pdb; pdb.set_trace()

    bestind = np.argmax(res['lnlik'])
    bestpars = res['pars'][bestind,:]
    bestmodel = ts.model(bestpars)
        
    #import pdb; pdb.set_trace()
        
    return res,bestpars,bestmodel
