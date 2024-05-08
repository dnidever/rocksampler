import os
import numpy as np
from dlnpyutils import utils as dln,coords

from spacerocks import SpaceRock, Units

from astroquery.jplhorizons import Horizons

from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord
from . import orbelem

units = Units()
units.timescale = 'tdb' # JPL gives elements in TDB. This actually matters for precise ephemerides.
#units.current()

# speed of light in AU/day
c_au_per_day = 173.14463267

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

class EarthRock(object):

    def __init__(self):
        t0 = Time('2000',format='byear').jd
        rock = SpaceRock(q=0.98327,
                         e=0.01671022,
                         inc=0.00005,
		         node=-11.26064,
                         arg=102.94719,
		         M=8.7777999,
                         epoch=t0,
                         name='earth',
                         origin='ssb',
                         frame='eclipJ2000',
                         units=units)
        self._data = rock

    def __call__(self,t):
        """ Get the position at a time JD time."""
        t = np.atleast_1d(np.array(t))
        rock = self._data
        out = []
        for i in range(len(t)):
            #prop,planets,sim = prop.propagate(epochs=t[i],progress=False)
            # analytic_propagate() preserves the orbital elements
            rock = rock.analytic_propagate(t[i])
            pos = (rock.x[0].to(u.AU).value,
                   rock.y[0].to(u.AU).value,
                   rock.z[0].to(u.AU).value)
            out.append(pos)
        self._data = rock
        if len(t)==1:
            out = out[0]
        else:
            # reshape
            xout = [p[0] for p in out]
            yout = [p[1] for p in out]
            zout = [p[2] for p in out]
            out = (xout,yout,zout)
        return out

    @property
    def a(self):
        """ Semi-major axis in AU """
        return self._data.a[0].to(u.AU).value

    @property
    def e(self):
        """ Eccentricity """
        return self._data.e[0]

    @property
    def inc(self):
        """ Inclination """
        return self._data.inc[0].deg

    @property
    def node(self):
        """ Longitude of the ascending Node in deg """
        return self._data.node[0].deg

    @property
    def arg(self):
        """ Argument of periapsis/perihelion (omega) in deg """
        return self._data.arg[0].deg

    @property
    def M(self):
        """ Mean anomaly at epoch in deg """
        return self._data.M[0].deg
    
    @property
    def epoch(self):
        return self._data.epoch[0].jd

    @property
    def elems(self):
        """ Orbital elements."""
        return self.a,self.e,self.inc,self.node,self.arg,self.M,self.epoch
    
    @property
    def position(self):
        """ Return position in AU."""
        out = (self._data.x[0].to(u.AU).value,
               self._data.y[0].to(u.AU).value,
               self._data.z[0].to(u.AU).value)
        return out

    @property
    def velocity(self):
        """ Return position in AU."""
        out = (self._data.vx[0].to(u.AU/u.day).value,
               self._data.vy[0].to(u.AU/u.day).value,
               self._data.vz[0].to(u.AU/u.day).value)
        return out

    @property
    def state(self):
        """ Return state in AU and AU/day."""
        return self.position + self.velocity
    
    def obs2xyz(self,ra,dec,distance,epoch):
        """ Get barycentric xyz from ra/dec/distance/epoch."""
        # Get position relative to earth
        coo = SkyCoord(ra=ra*u.deg,dec=dec*u.deg,distance=distance*u.AU,
                       frame='icrs')
        ecoo = coo.transform_to('geocentricmeanecliptic')
        xobs,yobs,zobs = (ecoo.cartesian.x.to(u.AU).value,
                          ecoo.cartesian.y.to(u.AU).value,
                          ecoo.cartesian.z.to(u.AU).value)
        # Earth position
        xearth,yearth,zearth = self(epoch)
        # Barycentric positions
        x = xobs + xearth
        y = yobs + yearth
        z = zobs + zearth
        return x,y,z

    def obs2state(self,ra,dec,distance,epoch):
        """ Convert observed quantities to state vector """
        x,y,z = self.obs2xyz(ra,dec,distance,epoch)
        mnt = np.mean(epoch)
        ind1 = np.argmin(epoch)
        ind2 = np.argmax(epoch)
        dt = epoch[ind2]-epoch[ind1]
        vx = (x[ind2]-x[ind1])/dt
        vy = (y[ind2]-y[ind1])/dt
        vz = (z[ind2]-z[ind1])/dt
        # in AU per day
        state = [np.mean(x),np.mean(y),np.mean(z),vx,vy,vz]
        return state,mnt
    
    def obs2elem(self,ra,dec,distance,epoch):
        """ Use observed quantities to calculate orbital elements """
        state,mnt = self.obs2state(ra,dec,distance,epoch)
        elem = orbelem.calc_classical_elements(state,mnt)
        # a, e, inc, Omega (node),omega (argument), M, epoch
        out = (elem.major_axis,elem.ecc,elem.incl,
               elem.asc_node,elem.arg_per,elem.mean_anomaly,elem.epoch)
        return out

class RockSampler(object):

    def __init__(self):
        self._earth = EarthRock()

    def obs2xyz(self,ra,dec,distance,epoch):
        """ Get barycentric xyz from ra/dec/distance/epoch."""
        # Get position relative to earth
        coo = SkyCoord(ra=ra*u.deg,dec=dec*u.deg,distance=distance*u.AU,
                       frame='icrs')
        ecoo = coo.transform_to('geocentricmeanecliptic')
        xobs,yobs,zobs = (ecoo.cartesian.x.to(u.AU).value,
                          ecoo.cartesian.y.to(u.AU).value,
                          ecoo.cartesian.z.to(u.AU).value)
        # Earth position
        xearth,yearth,zearth = self._earth(epoch)
        # Barycentric positions
        x = xobs + xearth
        y = yobs + yearth
        z = zobs + zearth
        return x,y,z

    def obs2state(self,ra,dec,distance,epoch):
        """ Convert observed quantities to state vector """
        x,y,z = self.obs2xyz(ra,dec,distance,epoch)
        mnt = np.mean(epoch)
        ind1 = np.argmin(epoch)
        ind2 = np.argmax(epoch)
        dt = epoch[ind2]-epoch[ind1]
        vx = (x[ind2]-x[ind1])/dt
        vy = (y[ind2]-y[ind1])/dt
        vz = (z[ind2]-z[ind1])/dt
        # in AU per day
        state = [np.mean(x),np.mean(y),np.mean(z),vx,vy,vz]
        return state,mnt
    
    def obs2elem(self,ra,dec,distance,epoch):
        """ Use observed quantities to calculate orbital elements """
        state,mnt = self.obs2state(ra,dec,distance,epoch)
        elem = orbelem.calc_classical_elements(state,mnt)
        # a, e, inc, Omega (node),omega (argument), M, epoch
        out = (elem.major_axis,elem.ecc,elem.incl,
               elem.asc_node,elem.arg_per,elem.mean_anomaly,elem.epoch)
        return out

    def reasonableorbit(self,elem,epoch=None):
        """ Is this a reasonable orbit """
        if len(elem)==7:
            a,e,i,Omega,omega,M,epoch = elem
        elif len(elem)==6:
            a,e,i,Omega,omega,M = elem
        else:
            raise ValueError('elem must have 6 or 7 values')
        el = orbelem.Elements()
        el.major_axis = a
        el.ecc = e
        el.incl = i
        el.asc_node = Omega
        el.arg_per = omega
        el.mean_anomaly = M
        el.epoch = epoch
        state = el.state
        rvec = state[:3]
        vvec = state[3:]
        # current velocity and distance
        r = np.linalg.norm(rvec)
        v = np.linalg.norm(vvec)

        # flag = True for a reasonable orbit
        # flag = False for an unreasonable orbit
        flag = True  # good to start
        if a<0:
            flag = False
        elif np.abs(a)<1e-5:
            flag = False
        if ~np.isfinite(r):
            flag = False
        if ~np.isfinite(v):
            flag = False
        if r<1e-5:     # too small
            flag = False
        if r>10000:    # too far away
            flag = False
        if v<1e-5:     # too small
            flag = True
        if v>0.05*c_au_per_day:  # too fast
            flag = False
        
        return flag

def sampler(tab,nsamples=10000):
    """ Sampler for tracklet. """

    # Run lots of samples on the distance of the first tracklet measurement
    # and the distance of the second tracklet measurement
    # See if any of those give reasonable orbits

    # Perform rejection sampling

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

    # Start the RockSampler object
    rs = RockSampler()

    # Want first and last observations
    ind1 = np.argmin(t)
    ind2 = np.argmax(t)
    rr = [ra[ind1],ra[ind2]]
    dd = [dec[ind1],dec[ind2]]
    tt = [t[ind1],t[ind2]]
    dtime = tt[1]-tt[0]  # in days

    # speed of light is 1 AU in 8 min
    # max disance is 5% of light speed
    maxdist = dtime/(0.05*c_au_per_day)
    
    r1range = [-3,2]  # log
    dr2range = [-maxdist,maxdist]
    
    # Create the samples
    r1samples = 10**np.random.uniform(r1range[0],r1range[1],nsamples)
    dr2samples = np.random.uniform(dr2range[0],dr2range[1],nsamples)
    
    dt = [('r1',float),('r2',float),('valid',bool),('elem',float,6),
          ('state',float,6),('r',float),('v',float)]
    res = np.zeros(nsamples,dtype=np.dtype(dt))

    # Could do a while loop until we have enough valid orbits
    
    # Sample loop
    for i in range(nsamples):
        if i % 500 == 0: print(i)
        r1 = r1samples[i]
        dr2 = dr2samples[i]
        r2 = r1 + dr2
        r2 = np.maximum(r2,0.001)  # make sure it's positive
        dist = [r1,r2]
        try:
            elem = rs.obs2elem(rr,dd,dist,tt)
            state = orbelem.calc_state(elem[:6],elem[6])
            valid = rs.reasonableorbit(elem)
        except KeyboardInterrupt:
            return            
        except:
            valid = False
            elem = (np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan)
            state = (np.nan,np.nan,np.nan,np.nan,np.nan,np.nan)
        res['r1'][i] = r1
        res['r2'][i] = r2
        res['valid'][i] = valid
        res['elem'][i] = elem[:6]
        res['state'][i] = state
        res['r'][i] = np.linalg.norm(state[:3])
        res['v'][i] = np.linalg.norm(state[3:])
        #print(i,valid)
        #print('  ',elem)

        #import pdb; pdb.set_trace()

    gd, = np.where(res['valid'])
    vres = res[gd]
        
    #import pdb; pdb.set_trace()

    return vres
    
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
