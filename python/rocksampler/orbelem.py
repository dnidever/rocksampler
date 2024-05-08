import os
import numpy as np
import math


#   2011 Aug 11:  in dealing with exactly circular orbits,  and those
#   nearly exactly circular,  I found several loss of precision problems
#   in the angular elements (which become degenerate for e=0;  you can
#   add an arbitrary amount to the mean anomaly,  as long as you subtract
#   the same amount from the argument of periapsis.)  Also,  q was
#   computed incorrectly for such cases when roundoff error resulted in
#   the square root of a number that should be zero,  but rounded below
#   zero,  was taken.  All this is fixed now.
#
#   2009 Nov 24:  noticed a loss of precision problem in computing arg_per.
#   This was done by computing the cosine of that value,  then taking the
#   arc-cosine.  But if that value is close to +/-1,  precision is lost
#   (you can actually end up with a domain error if the roundoff goes
#   against you).  I added code so that,  if |cos_arg_per| > .7,  we
#   compute the _sine_ of the argument of periapsis and use that instead.
#
#   While doing this,  I also noticed that several variables could be made
#   of type const.
#
#   calc_classical_elements( ) will take a given state vector r at a time t,
#   for an object orbiting a mass gm;  and will compute the orbital elements
#   and store them in the elem structure.  Normally,  ref=1.  You can set
#   it to 0 if you don't care about the angular elements (inclination,
#   longitude of ascending node,  argument of perihelion).
#
#  In determining the mean anomaly from the eccentricity and
# eccentric anomaly,  use of the "normal" formulae
#
# M = E - ecc * sin( E)         (elliptical case)
# M = E - ecc * sinh( E)        (hyperbolic case)
#
#   you run into nasty loss-of-precision problems with near-parabolic orbits:
# E and ecc * sin(E) will be nearly equal quantities.  In such cases,  it's
# better to use power series for the sin/sinh function, rearranging to get
#
# M = E(1-ecc) - ecc( -E^3/3! + E^5/5! - E^7/7!...)  (elliptical)
# M = E(1-ecc) - ecc(  E^3/3! + E^5/5! + E^7/7!...)  (hyperbolic)
#
#   ...in which the infinite series is that for sin/sinh,  minus
# the leading term E.  This can be expressed as
#
# M = E(1-ecc) - ecc * E * remaining_terms( -E^2)    (elliptical)
# M = E(1-ecc) - ecc * E * remaining_terms(  E^2)    (hyperbolic)
#

PI = 3.1415926535897932384626433832795028841971693993751058209749445923
SQRT_2 = 1.4142135623730950488016887242096980785696718753769480731766797

GAUSS_K = 0.01720209895
#SOLAR_GM = (GAUSS_K * GAUSS_K)
SOLAR_GM = 0.00029631   # spacerocks value, bary
# in AU^3/day^2
# GM_helio = 0.00029591220828411951
# GM_bary = 0.00029630927493457475


# G = 6.67430 x 10^-11 N m^2 / kg^2
# solar mass = 1.98847 x 10^30 kg

# GM =  kg N m^2 / kg^2 = (kg m/s^2) m^2 / kg = m^3/s^2

# in km^3/s^2  9 orders of mag difference, the values in these units are smaller by 1e9

# solar GM in km^3/s^2 is 1.327124381789709e+11

# 1 AU = 149,597,870,700 m = 1.49597 x 10^11 m
# earth radius = 6378 km
# jupiter radius = 69,911 km 
# solar radius = 695700 km

class Elements(object):

    def __init__(self):
        self.perih_time = 0.0
        self.q = 0.0
        self.Q = 0.0
        self.ecc = 0.0
        self.incl = 0.0
        self.arg_per = 0.0
        self.asc_node = 0.0
        self.epoch = 0.0
        self.mean_anomaly = 0.0
        # derived quantities
        self.period = 0.0
        self.lon_per = 0.0
        self.minor_to_major = 0.0
        self.perih_vec = np.zeros(3,float)
        self.sideways = np.zeros(3,float)
        self.angular_momentum = 0.0
        self.major_axis = 0.0
        self.t0 = 0.0
        self.w0 = 0.0
        self.abs_mag = 0.0
        self.slope_param = 0.0
        self.mean_motion = 0.0
        self.gm = 0.0
        self.is_asteroid = False
        self.central = False
        self.orbit_type = 0  # 1: elliptical, 2: parabolic, 3: hyperbolic
        self._state = None
        
        # double perih_time, q, ecc, incl, arg_per, asc_node;
        # double epoch,  mean_anomaly;
        #          /* derived quantities: */
        # double lon_per, minor_to_major;
        # double perih_vec[3], sideways[3];
        # double angular_momentum, major_axis, t0, w0;
        # double abs_mag, slope_param, gm;
        # int is_asteroid, central_obj;

    def __repr__(self):
        out = '<Elements ['
        vals = []
        cols = ['major_axis','ecc','incl','asc_node','arg_per','mean_anomaly','epoch']
        names = ['a','e','i','node','arg','M','t']
        for c,n in zip(cols,names):
            vals.append('{:}={:.3f}'.format(n,getattr(self,c)))
        out += ','.join(vals)
        out += ']>'
        return out

    @property
    def state(self):
        """ Return the state vector """
        if self._state is None:
            elem = [self.major_axis,self.ecc,self.incl,
                    self.asc_node,self.arg_per,self.mean_anomaly]
            state = calc_state(elem,self.epoch)
            self._state = state
        return self._state

def remaining_terms(ival):
    rval = 0.0
    z = 1.0
    tolerance = 1e-30
    i = 2

    while abs(z) > tolerance:
        z *= ival / (i * (i + 1))
        rval += z
        i += 2

    return rval

def calc_classical_elements(state, t, gm=SOLAR_GM, ref=True):
    """
    Calculate orbital elements given a state vector (position+velocity) and time.

    Parameters
    ----------
    state : numpy array or list
       State vector with 3 position and 3 velocity elements.
         The units should be in AU and days.
    t : float
       Time in days.
    gm : float, optional
       G*M value for the given system in units of AU and days.
         Default is Solar value.
    ref : bool, optional
       Set to False if you don't care about the angular momentum.

    Returns
    -------
    elem : Elements object
       Elements object with all of the orbital elements information.

    Examples
    --------

    state = [1.0, 0.0, 0.0,  0.0, 0.0172, 0.0]
    t = 2451545.0
    elem = calc_classical_elements(state,t)

    """
    
    # need gm, default to solar GM
    # r = [x, y, z, vx, vy, vz], state vector

    # Initalize Elements object
    elem = Elements()
    elem.gm = gm
    
    r = state[:3]
    v = state[3:]
    #r_dot_v = dot_product(r, v)
    r_dot_v = np.dot(r,v)
    #dist = vector3_length(r)
    dist = np.linalg.norm(r)
    #v2 = dot_product(v, v)
    v2 = np.dot(v,v)
    inv_major_axis = 2.0 / dist - v2 / elem.gm
    ecc2 = 0.0
    ecc = 0.0
    
    assert elem.gm != 0.0
    #h = vector_cross_product(r, v)
    h = np.cross(r,v)
    n0 = h[0] * h[0] + h[1] * h[1]
    h0 = n0 + h[2] * h[2]
    assert dist > 0.0
    assert v2 > 0.0      # elements are undefined if the object is at rest
    assert h0 > 0.0      # or if its velocity vector runs through the sun
    n0 = math.sqrt(n0)   # component of ang. mom. NOT in the plane
    h0 = math.sqrt(h0)   # specific angular momentum
    
    # See Danby,  p 204-206,  for much of this:
    
    if ref:
        # orbit is in xy plane;  asc node is undefined;  make
        # arbitrary choice h[0] = 0, h[1] = -epsilon
        if n0 == 0:
            elem.asc_node = 0.0
        else:
            elem.asc_node = math.atan2(h[0], -h[1])
            #elem.asc_node = math.pi/2 + elem.asc_node   # why??
        elem.incl = math.asin(n0 / h0)
        # retrograde orbit
        if h[2] < 0.0:
            elem.incl = math.pi - elem.incl
            
    #vector_cross_product(e, v, h)
    e = np.cross(v,h)
    for i in range(3):
        e[i] = e[i] / elem.gm - r[i] / dist
    # "flatten" e vector into the rv
    # plane to avoid roundoff; see
    # above comments
    #tval = dot_product(e, h) / h0
    tval = np.dot(e,h) / h0
    for i in range(3):
        e[i] -= h[i] * tval
    #ecc2 = dot_product(e, e)
    ecc2 = np.dot(e,e)
    # avoid roundoff issues w/nearly parabolic orbits
    if math.fabs(ecc2 - 1.0) < 1.e-14:
        ecc2 = 1.0
    elem.minor_to_major = math.sqrt(math.fabs(1.0 - ecc2))
    ecc = elem.ecc = math.sqrt(ecc2)

    # for purely circular orbits,  e is
    # arbitrary in the orbit plane;
    # choose r normalized
    if ecc == 0:
        for i in range(3):
            e[i] = r[i] / dist
    # ...and if it's not circular,
    # normalize e: 
    else:
        for i in range(3):
            e[i] /= ecc
    if ecc < 0.9:
        elem.q = (1.0 - ecc) / inv_major_axis
    else:
        # at eccentricities near one,  the above suffers
        # a loss of precision problem,  and we switch to:
        gm_over_h0 = elem.gm / h0
        perihelion_speed = gm_over_h0 * (1.0 + math.sqrt(1.0 - inv_major_axis * h0 * h0 / elem.gm))
        assert h0 != 0.0
        assert gm_over_h0 != 0.0
        assert math.isfinite(inv_major_axis)
        assert math.isfinite(gm_over_h0)
        assert math.isfinite(perihelion_speed)
        assert perihelion_speed != 0.0
        elem.q = h0 / perihelion_speed
        assert elem.q != 0.0                     # For q=0,  nothing is defined
        inv_major_axis = (1.0 - ecc) / elem.q
    
    assert elem.q != 0.0         # For q=0,  nothing is defined 
    assert elem.q > 0.0
    
    if inv_major_axis:
        elem.major_axis = 1.0 / inv_major_axis
        elem.t0 = elem.major_axis * math.sqrt(math.fabs(elem.major_axis) / elem.gm)
    
    #vector_cross_product(elem.sideways, h, e)
    elem.sideways = np.cross(h,e)
    if ref:
        if n0:
            cos_arg_per = (h[0] * e[1] - h[1] * e[0]) / n0
        else:
            cos_arg_per = e[0]
        if 0.7 > cos_arg_per > -0.7:
            elem.arg_per = math.acos(cos_arg_per)
        else:
            if n0:
                sin_arg_per = (e[0] * h[0] * h[2] + e[1] * h[1] * h[2] - e[2] * n0 * n0) / (n0 * h0)
            else:
                sin_arg_per = e[1] * h[2] / h0
            elem.arg_per = math.fabs(math.asin(sin_arg_per))
            if cos_arg_per < 0.0:
                elem.arg_per = math.pi - elem.arg_per
        if e[2] < 0.0:
            elem.arg_per = math.pi + math.pi - elem.arg_per

    elem.epoch = t
            
    # elliptical orbit
    if ecc>=0 and ecc<1:
        elem.orbit_type = 1  # ellipse
    elif ecc==1:
        elem.orbit_type = 2  # parabolic
    elif ecc>1:
        elem.orbit_type = 3  # hyperbolic
    else:
        elem.orbit_type = 0  # ??
            
    if inv_major_axis and elem.minor_to_major:
        is_nearly_parabolic = 0.99999 < ecc < 1.00001
        #r_cos_true_anom = dot_product(r, e)
        r_cos_true_anom = np.dot(r, e)
        #r_sin_true_anom = dot_product(r, elem.sideways) / h0
        r_sin_true_anom = np.dot(r, elem.sideways) / h0
        sin_E = r_sin_true_anom * inv_major_axis / elem.minor_to_major
        assert elem.minor_to_major
        assert math.isfinite(ecc)
        assert math.isfinite(h0)
        assert math.isfinite(r_cos_true_anom)
        assert math.isfinite(r_sin_true_anom)
        assert math.isfinite(sin_E)
        # parabolic case
        if inv_major_axis > 0.0:
            cos_E = r_cos_true_anom * inv_major_axis + ecc
            ecc_anom = math.atan2(sin_E, cos_E)
            assert math.isfinite(cos_E)
            assert math.isfinite(ecc_anom)
            if is_nearly_parabolic:
                elem.mean_anomaly = ecc_anom * (1 - ecc) - ecc * ecc_anom * remaining_terms(-ecc_anom * ecc_anom)
            else:
                elem.mean_anomaly = ecc_anom - ecc * sin_E
            assert math.isfinite(elem.mean_anomaly)
            elem.perih_time = t - elem.mean_anomaly * elem.t0
        # hyperbolic case
        else:
            ecc_anom = math.asinh(sin_E)
            if is_nearly_parabolic:
                elem.mean_anomaly = ecc_anom * (1 - ecc) - ecc * ecc_anom * remaining_terms(ecc_anom * ecc_anom)
            else:
                elem.mean_anomaly = ecc_anom - ecc * sin_E
            assert math.isfinite(elem.mean_anomaly)
            assert elem.t0 <= 0.0
            elem.perih_time = t - elem.mean_anomaly * math.fabs(elem.t0)
            h0 = -h0
    # parabolic case
    else:
        tau = math.sqrt(dist / elem.q - 1.0)
        if r_dot_v < 0.0:
            tau = -tau
        elem.w0 = (3.0 / math.sqrt(2)) / (elem.q * math.sqrt(elem.q / elem.gm))
        elem.perih_time = t - tau * (tau * tau / 3.0 + 1) * 3.0 / elem.w0
        
    # At this point,  elem.sideways has length h0.  Make it a unit vect:
    for i in range(3):
        elem.perih_vec[i] = e[i]
        elem.sideways[i] /= h0
    elem.angular_momentum = h0

    # Calculate the period (in days)
    # P^2 = 4*pi^2 * a^3 / (G*M)
    elem.period = np.sqrt( 4*np.pi**2 * elem.major_axis**3 / elem.gm)

    # Q, apoapsis_dist
    elem.Q = 2*elem.major_axis - elem.q

    # Mean motion
    elem.mean_motion = np.sqrt(elem.gm/elem.major_axis**3)
    
    # Convert to degrees
    elem.incl = np.rad2deg(elem.incl)
    elem.asc_node = np.rad2deg(elem.asc_node)
    elem.arg_per = np.rad2deg(elem.arg_per)
    elem.mean_anomaly = np.rad2deg(elem.mean_anomaly)
    if elem.mean_anomaly < 0:
        elem.mean_anomaly += 360
            
    return elem

def M_to_E(M,e):
    """
    Calculate E (eccentric anomaly) from M (mean anomaly).
    https://en.wikipedia.org/wiki/Elliptic_orbit
    Kepler's equation: M = E - e*sin(E)

    Parameters
    ----------
    M : float
       Mean anomaly in degrees.
    e : float
       Eccentricity.

    Returns
    -------
    E : float
       Eccentric anomaly in degrees.
    
    Examples
    --------

    E = M_to_E(100.0,0.1)

    """

    # https://www.johndcook.com/blog/2021/04/01/efficient-kepler-equation/#:~:text=Efficiently%20solving%20Kepler's%20equation&text=M%20%2B%20e%20sin%20E%20%3D%20E,solve%20for%20eccentric%20anomaly%20E.&text=and%20take%20an%20initial%20guess,f(E)%20%3D%20E.
    # There is a simple way to solve this equation. Define
    # 
    # f(E) = M + e sin E
    # 
    # and take an initial guess at the solution and stick it into f. Then take
    # the output and stick it back into f, over and over, until you find a
    # fixed point, i.e. f(E) = E.

    count = 0
    flag = False
    Mrad = np.deg2rad(M)
    Erad = Mrad
    last_Erad = Erad
    while (flag==False):
        Erad = Mrad + e*np.sin(Erad)
        diff = np.abs(Erad-last_Erad)
        last_Erad = Erad
        count += 1
        if count > 10 or diff<1e-10:
            flag = True
              
    E = np.rad2deg(Erad)
    return E
        
def calc_state(elem, t, gm=SOLAR_GM):
    """
    Calculate the state vector from the orbital elements and epoch.

    Parameters
    ----------
    elem : numpy array or list
       Orbital elements:
           a : semi-major axis (in AU)
           e : eccentricity
           inc : inclination (in deg)
           Omega : longitude of ascending node (in deg)
           omega : argument of perihelion (in deg)
           M : mean anomaly (in deg)
    t : float
       JD Epoch in days.
    gm : float, optional
       The standard gravitational parameters (G*M) in AU^3/day^2

    Returns
    -------
    state : numpy array
       Barycentric cartesians position and velocities in AU and AU/day.
         [x, y, z, vx, vy, vz].

    Examples
    --------

    state = calc_state(elem,t)

    """
    # https://orbital-mechanics.space/classical-orbital-elements/orbital-elements-and-the-state-vector.html

    # need h (ang momentum), e, nu (true anomaly), mu (gm),
    #      omega (argument), Omega (node), inc
    
    a,e,inc,Omega,omega,M = elem
    mu = gm

    # r_peri = h**2/mu * 1/(1+e)
    rperi = a*(1-e)
    h = np.sqrt(mu * rperi * (1+e))
    
    # from mean anomaly to eccentric anomaly
    # https://en.wikipedia.org/wiki/Elliptic_orbit
    # Kepler's equation: M = E - e*sin(E)
    E = M_to_E(M,e)  # in degree
    # from eccentric anomaly to true anomaly
    # https://en.wikipedia.org/wiki/True_anomaly
    nu = np.arccos((np.cos(np.deg2rad(E))-e)/(1-e*np.cos(np.deg2rad(E))))
    # in radians
    
    # Step 1: Transform to Perifocal Frame
    r_w = h**2 / mu / (1 + e * np.cos(nu)) * np.array((np.cos(nu), np.sin(nu), 0))
    v_w = mu / h * np.array((-np.sin(nu), e + np.cos(nu), 0))

    # Step 2: Rotate the perifocal plane
    from scipy.spatial.transform import Rotation

    R = Rotation.from_euler("ZXZ", [-np.deg2rad(omega), -np.deg2rad(inc), -np.deg2rad(Omega)])    
    r_rot = r_w @ R.as_matrix()
    v_rot = v_w @ R.as_matrix()

    state = np.concatenate((r_rot,v_rot))

    return state
