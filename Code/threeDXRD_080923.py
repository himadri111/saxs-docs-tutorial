"""
13.06.21: remove np.vectorize statements
13.06.2021: done
"""
import numpy as np
from mayavi import mlab
from scipy.integrate import quad
from time import time
from numba import jit, vectorize, guvectorize, float32, float64, int64, int32, int16
import itertools

from scipy.special import expit

def normFac(f0,df0,dq):
    
    if (f0==0):
        fval = (1./(2*np.pi)**1.5)*(1.0/(df0*df0))*(1.0/dq)
        #print(fval)
    else:
        fval = (1./(2*np.pi)**2)*(1./np.sin(f0))*(1.0/df0)*(1.0/dq)
        
    return fval

def qxyz_det(q,chi,wavelen):
    """
    Parameters
    ----------
    q : float
        scattering wavevector.
    chi : float
        angle on 2D detector.    
    wavelen : float
        X-ray wavelength.

    Returns
    -------
    float tuple of (qx, qy, qz) corresponding to reciprocal space
    point where I(q,chi) is sampled; X-ray wavelength = wavelen
    Config: beam along qy, qx vertical, qz horizontal
    Draw the Ewald sphere intersection construction to see how 
    equations come about
    """
    qE = (2*np.pi)/wavelen
    qx = q * np.sqrt(1-(q/(2*qE))**2) * np.sin(chi)
    qy = -q**2/(2*qE)
    qz = q * np.sqrt(1-(q/(2*qE))**2) * np.cos(chi)
    return qx, qy, qz

#@jit(nopython=True)
def Theta(qx, qy, qz, gamma):
    """
    Parameters
    ----------
    qx : float
        x-component wavevector, nm^-1.
    qy : float
        y-component wavevector, nm^-1.
    qz : float
        z-component wavevector, nm^-1.
    gamma : float
        tilt angle of fibre in body-fixed qx-qz plane.
    Returns
    -------
    fval : float
        arctan argument used in the angle dependent part
        of the scattering function
    """
    q = np.sqrt(qx**2 + qy**2 + qz**2)
    fval = np.zeros_like(qx)
    fval = np.where(np.logical_and((np.abs((qx*np.cos(gamma)+qy*np.sin(gamma))**2+qz**2)>0), \
                    ((q**2/((qx*np.cos(gamma)+qy*np.sin(gamma))**2+qz**2))-1)>=0),\
                    np.arctan(np.sqrt((q**2/((qx*np.cos(gamma)+qy*np.sin(gamma))**2+qz**2))-1)),\
                        np.pi/2.0)
    """
    Added term in logical_and to check the sqrt term is non-negative
    
    if np.abs((qx*np.cos(gamma)+qy*np.sin(gamma))**2+qz**2)>0:
        s2 = q**2/((qx*np.cos(gamma)+qy*np.sin(gamma))**2+qz**2)
        fval = np.arctan(np.sqrt(s2-1))
    else: # the divergence of s2 indicates +infty -> pi/2 for arctan()
        fval = np.pi/2.0
    """
    return fval

#@jit(nopython=True)
def tpositive(q0, q, qx, qy, qz):
    
    #qperp = np.sqrt(qx**2+qz**2)
    t = q/np.sqrt((qy+q-q0)**2+(qx**2+qz**2))
    return t

#@jit(nopython=True)
def tnegative(q0, q, qx, qy, qz):
    
    #qperp = np.sqrt(qx**2+qz**2)
    t = q/np.sqrt((qy-q+q0)**2+(qx**2+qz**2))
    return t

#@jit(nopython=True)
def qxqypositive(q0, q, qx, qy, qz, t):
    qxt = t*qx
    qyt = t*qy - (1-t)*(q-q0)
    qzt = t*qz
    return qxt, qyt, qzt

#@jit(nopython=True)
def qxqynegative(q0, q, qx, qy, qz, t):
    qxt = t*qx
    qyt = t*qy + (1-t)*(q-q0)
    qzt = t*qz
    return qxt, qyt, qzt

#@jit(nopython=True)
def Isinglefibril_intermediate(qx,qy,qz,gamma,mu,q0,wMu,deltaQ0,Normfac,delta=0.5):
    """
    Parameters
    ----------
    qx : float
        x-component wavevector, nm^-1.
    qy : float
        y-component wavevector, nm^-1.
    qz : float
        z-component wavevector, nm^-1.
    gamma : float
        tilt angle of fibre in body-fixed qx-qz plane.
    mu : float
        In flat ellipsoid case, mu should be zero
        Put an assert statement to ensure this is the case
        angle of reciprocal lattice vector wrt fibre axis.
    q0 : float
        magnitude of reciprocal lattice vector.
    wMu : float
        for flat ellipsoid, deltaQE should be q0 tan(wMu) and represents
        the equatorial width of the ellipsoid perpendicular to the fibril
    deltaQ0 : float
        radial width of reciprocal space torus in
        q-direction
    Normfac : float
        Normalising factor which is different depending on whether
        mu = 0 or not
    delta = parameter from 0 to 1: for 0, corresponds to flat case, for 1
    corresponds to spherical case
        
    NOTE: mu and gamma should be 0
    
    MAXVAL: a upper-threshold to suppress divergent expressions in integrand
    
    Returns
    -------
    float
        scattering intensity for mu_th reciprocal lattice
        node for a fibre symmetric material, at (qx,qy,qz).

    """
    #delta=0.5
    q = q0/delta
    
    MAXVAL=1.0e+8
    muBar = np.pi/2.0-mu
    qabs = np.sqrt(qx**2 + qy**2 + qz**2)
    deltaQE = q0 * np.tan(wMu) # this is wp
    prefactor = (1.0/(np.sqrt(2*np.pi)*deltaQ0))*(1.0/(np.sqrt(2*np.pi)*deltaQE))
    #prefactor = 1.0
    
    #calculate intersection point
    tplus = np.where(qy>0,tpositive(q0,q,qx,qy,qz),tnegative(q0,q,qx,qy,qz))
    qxi, qyi, qzi = np.where(qy>0,qxqypositive(q0, q, qx, qy, qz, tplus),qxqynegative(q0, q, qx, qy, qz, tplus))
    chiplus = np.arcsin(tplus*np.sqrt(qx**2+qz**2)/q)
    
    dmetric = np.sqrt((qx-qxi)**2+(qy-qyi)**2+(qz-qzi)**2)
    #print(tplus)
    #print(qy)
    #print(chiplus)
    #print(dmetric)
    chimetric = q*chiplus
    #print(chimetric)
    
    expfactor1 = np.exp(-0.5*((chimetric**2)/(deltaQE**2)))
    #expfactor2 = np.exp(-0.5*(dmetric/deltaQ0)**2)
    expfactor2 = 1/(1+(dmetric/deltaQ0)**2)
    
    #print("deltaQ0: ", deltaQ0, " deltaQE: ", deltaQE)
    # this is the equatorial width of the streak, like deltaQ0 is the 
    # radial width of the streak; also called wp
    # expfactor1 = np.exp(-0.5*((qx**2+qz**2)/(deltaQE**2)))
    # expfactor2 = np.exp(-0.5*((qy-q0)/deltaQ0)**2)+np.exp(-0.5*((qy+q0)/deltaQ0)**2)
    #expfactor1 = np.exp(-0.5*((qx**2+qz**2)/(deltaQE**2)))
    #expfactor2 = np.exp(-0.5*((qy-q0)/deltaQ0)**2)+np.exp(-0.5*((qy+q0)/deltaQ0)**2)
    
    fval = prefactor * expfactor1 * expfactor2
    
    rval = np.zeros_like(fval)
    rval = np.where(fval<MAXVAL,fval,MAXVAL)
    """
    if fval < MAXVAL:
        return fval
    else:
        return MAXVAL
    """
    return rval

#@jit(nopython=True)
def Isinglefibril_flat(qx,qy,qz,gamma,mu,q0,wMu,deltaQ0,Normfac):
    """
    Parameters
    ----------
    qx : float
        x-component wavevector, nm^-1.
    qy : float
        y-component wavevector, nm^-1.
    qz : float
        z-component wavevector, nm^-1.
    gamma : float
        tilt angle of fibre in body-fixed qx-qz plane.
    mu : float
        In flat ellipsoid case, mu should be zero
        Put an assert statement to ensure this is the case
        angle of reciprocal lattice vector wrt fibre axis.
    q0 : float
        magnitude of reciprocal lattice vector.
    wMu : float
        for flat ellipsoid, deltaQE should be q0 tan(wMu) and represents
        the equatorial width of the ellipsoid perpendicular to the fibril
    deltaQ0 : float
        radial width of reciprocal space torus in
        q-direction
    Normfac : float
        Normalising factor which is different depending on whether
        mu = 0 or not
        
    MAXVAL: a upper-threshold to suppress divergent expressions in integrand
    
    Returns
    -------
    float
        scattering intensity for mu_th reciprocal lattice
        node for a fibre symmetric material, at (qx,qy,qz).

    """
    MAXVAL=1.0e+8
    muBar = np.pi/2.0-mu
    qabs = np.sqrt(qx**2 + qy**2 + qz**2)
    prefactor = Normfac
    deltaQE = q0 * np.tan(wMu) 
    #print("deltaQ0: ", deltaQ0, " deltaQE: ", deltaQE)
    # this is the equatorial width of the streak, like deltaQ0 is the 
    # radial width of the streak; also called wp
    expfactor1 = np.exp(-0.5*((qx**2+qz**2)/(deltaQE**2)))
    expfactor2 = np.exp(-0.5*((qy-q0)/deltaQ0)**2)+np.exp(-0.5*((qy+q0)/deltaQ0)**2)
    fval = prefactor * expfactor1 * expfactor2
    
    rval = np.zeros_like(fval)
    rval = np.where(fval<MAXVAL,fval,MAXVAL)
    """
    if fval < MAXVAL:
        return fval
    else:
        return MAXVAL
    """
    return rval

#@jit(nopython=True)
def Isinglefibril(qx,qy,qz,gamma,mu,q0,wMu,deltaQ0,Normfac):
    """
    Parameters
    ----------
    qx : float
        x-component wavevector, nm^-1.
    qy : float
        y-component wavevector, nm^-1.
    qz : float
        z-component wavevector, nm^-1.
    gamma : float
        tilt angle of fibre in body-fixed qx-qz plane.
    mu : float
        angle of reciprocal lattice vector wrt fibre axis.
    q0 : float
        magnitude of reciprocal lattice vector.
    wMu : float
        angular width of reciprocal space torus in
        qx-qz plane.
    deltaQ0 : float
        radial width of reciprocal space torus in
        q-direction
    Normfac : float
        Normalising factor which is different depending on whether
        mu = 0 or not
        
    MAXVAL: a upper-threshold to suppress divergent expressions in integrand
    
    Returns
    -------
    float
        scattering intensity for mu_th reciprocal lattice
        node for a fibre symmetric material, at (qx,qy,qz).

    """
    MAXVAL=1.0e+8
    muBar = np.pi/2.0-mu
    qabs = np.sqrt(qx**2 + qy**2 + qz**2)
    prefactor = Normfac
    expfactor1 = np.exp(-0.5*((Theta(qx, qy, qz, gamma)-muBar)/wMu)**2)
    expfactor2 = np.exp(-0.5*((qabs-q0)/deltaQ0)**2)
    fval = prefactor * expfactor1 * expfactor2
    
    rval = np.zeros_like(fval)
    rval = np.where(fval<MAXVAL,fval,MAXVAL)
    """
    if fval < MAXVAL:
        return fval
    else:
        return MAXVAL
    """
    return rval

def w3D(theta,A,B):
    """
    2D planar fibre distribution, Onsager form
    Parameters
    ----------
    gamma : float
        tilt angle.
    gamma0 : float
        centre of Onsager distribution (Onsager 1949) in gamma.
    deltaGamma0 : float
        related to FWHM of Onsager distribution.

    Returns
    -------
    fval : float
        value of Onsager weighting function.

    """
    fval  = A*np.exp(-B*np.sin(theta)**2)
    return fval

#@jit(nopython=True)
def wgamma_ons(gamma,gamma0,deltaGamma0):
    """
    2D planar fibre distribution, Onsager form
    Parameters
    ----------
    gamma : float
        tilt angle.
    gamma0 : float
        centre of Onsager distribution (Onsager 1949) in gamma.
    deltaGamma0 : float
        related to FWHM of Onsager distribution.

    Returns
    -------
    fval : float
        value of Onsager weighting function.

    """
    kappa = np.log(2.0)/(np.sin(deltaGamma0))**2
    fval  = np.exp(-kappa*(np.sin(gamma-gamma0)**2))
    return fval

#@jit(nopython=True)
def wgamma(gamma,gamma0,deltaGamma0):
    """
    2D planar fibre distribution, default form
    Parameters
    ----------
    gamma : float
        tilt angle.
    gamma0 : float
        centre of distribution in gamma.
    deltaGamma0 : float
        FWHM .

    Returns
    -------
    fval : float
        value of weighting function.

    """
    fval  = np.exp(-0.5*(((gamma-gamma0)/deltaGamma0)**2))
    #return fval*kappa
    return fval

#@jit(nopython=True)
def Iplanarfibril_integrand(gamma,qx,qy,qz,gamma0,deltaGamma0,mu,q0,wMu,\
                            deltaQ0,Normfac,flat=0,delta=0.5):
    """
    Parameters
    ----------
    qx : float
        x-component wavevector, nm^-1.
    qy : float
        y-component wavevector, nm^-1.
    qz : float
        z-component wavevector, nm^-1.
    gamma : float
        tilt angle of fibril.
    gamma0 : float
        centre of Onsager distribution.
    deltaGamma0 : float
        related to FWHM of weighting planar distribution.
    mu : float
        angle of reciprocal lattice vector wrt fibre axis.
    q0 : float
        magnitude of reciprocal lattice vector.
    wMu : float
        angular width of reciprocal space torus in
        qx-qz plane.
    deltaQ0 : float
        radial width of reciprocal space torus in
        q-direction.
    Normfac : float
        Normalising factor which is different depending on whether
        mu = 0 or not

    Returns
    -------
    float
        integrand in scattering function.

    """
    
    """
    f1 = Isinglefibril(qx,qy,qz,gamma,mu,q0, wMu,deltaQ0,Normfac)
    f2 = wgamma(gamma,gamma0,deltaGamma0)
    print("shape 1: ", np.shape(f1)," shape 2: ", np.shape(f2))
    fv = np.multiply.outer(f1,f2)
    return fv
    """
    if (flat==0):
        return wgamma(gamma,gamma0,deltaGamma0)*Isinglefibril(qx,qy,qz,gamma,mu,q0, wMu,deltaQ0,Normfac)
    else:
        return wgamma(gamma,gamma0,deltaGamma0)*Isinglefibril_flat(qx,qy,qz,gamma,mu,q0, wMu,deltaQ0,Normfac)
        
"""
@vectorize([float64(float64, float64, float64, float64, float64, float64, \
                    float64, float64, float64, float64, int64, int64, float64)])
"""
def DiscreteFibrilIntegral(qx,qy,qz,gamma0,deltaGamma0,mu,q0,wMu,deltaQ0,Normfac,N,flat=0,delta=0.5):
    #gr = np.linspace(gamma0-deltaGamma0,gamma0+deltaGamma0,N)
    gr = np.linspace(gamma0-0.975*np.pi/2.0,gamma0+0.975*np.pi/2.0,N)
        
    wr = wgamma(gr,gamma0,deltaGamma0)
    
    """
    array_sum = np.sum(gr)
    #print("evaluating gr: ",np.isnan(array_sum))
    print("evaluating gr: ",np.isinf(array_sum))
    """
    fr = np.zeros_like(gr)
    #g=1.0
    fr = Iplanarfibril_integrand(gr,qx,qy,qz,gamma0,deltaGamma0,mu,q0,\
                                 wMu,deltaQ0,Normfac,flat)
    
    """
    array_sum = np.sum(fr)
    print("evaluating fr: ",np.isinf(array_sum))
    """
    
    fv = np.sum(wr*fr)/np.sum(wr)
    """
    #array_sum = np.sum(fv)
    #print("evaluating fv: ",np.isnan(array_sum))
    #print("evaluating fv: ",np.isinf(array_sum), " sum wr: ", np.sum(wr))
    """
    return fv

#@vectorize()
def Iplanarfibril(qxL,qyL,qzL,gamma0,deltaGamma0,mu,q0,wMu,deltaQ0,Normfac,\
                  alpha=0,beta=0,integrate=False,Ng=11,flat=0,delta=0.5):
    """ 
    Parameters
    ----------
    qx : float
        x-component wavevector, nm^-1.
    qy : float
        y-component wavevector, nm^-1.
    qz : float
        z-component wavevector, nm^-1.
    deltaGamma0 : float
        related to FWHM of weighting distribution.
    mu : float
        angle of reciprocal lattice vector wrt fibre axis.
    q0 : float
        magnitude of reciprocal lattice vector.
    wMu : float
        angular width of reciprocal space torus in
        qx-qz plane.
    deltaQ0 : float
        radial width of reciprocal space torus in
        q-direction.
    Normfac : float
        Normalising factor which is different depending on whether
        mu = 0 or not

    Returns
    -------
    float
        scattering function.
        
    13.06.2021: can we make this vectorized in code without having to call
    the np.vectorize argument?
    
    The usual components which are vectors are (qxL, qyL, qzL), which should 
    have the same shape
    
    Maybe we can create an array of the same shape as qxL(=qyL=qzL)
    
    13.06.2021: Yes, that works
    """
    Ng = 40 #number of points on the gamma function integrand
    #integralval = 1
    if (integrate==True):
        #print("should not be here")
        qx, qy, qz = rotated_vectors_rev(qxL,qyL,qzL,alpha,beta)
        integralval = np.zeros_like(qxL) #initialise an array
        """
        integralval = np.where(qx**2+qy**2+qz**2==0,0,\
                               quad(Iplanarfibril_integrand,-np.pi/2.0,np.pi/2.0,\
                                    args=(qx,qy,qz,gamma0,deltaGamma0,mu,q0,wMu,deltaQ0,Normfac),epsrel=1.0e-3)[0]) 
        """
        integralval = np.where(qx**2+qy**2+qz**2==0,0,\
                               DiscreteFibrilIntegral(qx,qy,qz,gamma0,deltaGamma0,\
                                      mu,q0,wMu,deltaQ0,Normfac,Ng,flat,delta=delta)) 
    elif (flat==0):
        """
        return intensity at peak of distribution gamma0
        useful if single fibril scattering wanted, as faster to evaluate
        """
        #print("should be here: ", flat)
        qx, qy, qz = rotated_vectors_rev(qxL,qyL,qzL,alpha,beta)
        integralval = np.zeros_like(qxL) #initialise an array
        integralval = np.where(qx**2+qy**2+qz**2==0,0,\
                               Isinglefibril(qx,qy,qz,gamma0,mu,q0,wMu,deltaQ0,Normfac))
    else:
        """
        return intensity at peak of distribution gamma0
        useful if single fibril scattering wanted, as faster to evaluate
        """
        #print("should be here - flat")
        qx, qy, qz = rotated_vectors_rev(qxL,qyL,qzL,alpha,beta)
        integralval = np.zeros_like(qxL) #initialise an array
        integralval = np.where(qx**2+qy**2+qz**2==0,0,\
                               Isinglefibril_intermediate(qx,qy,qz,gamma0,mu,q0,wMu,deltaQ0,Normfac,delta=delta))
                               #Isinglefibril_flat(qx,qy,qz,gamma0,mu,q0,wMu,deltaQ0,Normfac))
                               
   
    return integralval   

def multiplyRalpha(qx,qy,qz,alpha=0):
    Ralpha = np.array([[1,             0,            0],
                       [0, np.cos(alpha),np.sin(alpha)],
                       [0,-np.sin(alpha),np.cos(alpha)]])
    q = np.array([qx,qy,qz])
    qR = np.tensordot(Ralpha,q,axes=1)
    return qR[0],qR[1],qR[2]

def multiplyRbeta(qx,qy,qz,beta=0):
    Rbeta = np.array([[np.cos(beta),0,-np.sin(beta)],
                      [0,           1,            0],
                      [np.sin(beta),0,np.cos(beta)]])
    q = np.array([qx,qy,qz])
    #print(np.size(qx),np.size(qy),np.size(qz))
    qR = np.tensordot(Rbeta,q,axes=1)
    return qR[0],qR[1],qR[2]

def rotated_vectors(x,y,z,alpha=0,beta=0):
    """
    Parameters
    ----------
    x : float
        x-coordinate in body fixed frame.
    y : float
        y-coordinate in body fixed frame.
    z : float
        z-coordinate in body fixed frame.
    alpha : float, optional
        Rotation in radians around x-axis to reach body fixed frame from lab frame. 
        The default is 0.
    beta : float, optional
        Rotation in radians around y-axis to reach body fixed frame from lab frame.
        The default is 0.
        
    Returns
    -------
    xr : float
        x-coordinate in lab frame.
    yr : float
        y-coordinate in lab frame.
    zr : float
        z-coordinate in lab frame.
    """
    x1,y1,z1 = multiplyRalpha(x,y,z,-alpha)   # negative sign to rotate 
    xr,yr,zr = multiplyRbeta(x1,y1,z1,-beta)  # the fibril in lab frame 
    
    return xr, yr, zr

def rotated_vectors_rev(x,y,z,alpha=0,beta=0):
    """
    Parameters
    ----------
    x : float
        x-coordinate in body fixed frame.
    y : float
        y-coordinate in body fixed frame.
    z : float
        z-coordinate in body fixed frame.
    alpha : float, optional
        Rotation in radians around x-axis to reach body fixed frame from lab frame. 
        The default is 0.
    beta : float, optional
        Rotation in radians around y-axis to reach body fixed frame from lab frame.
        The default is 0.
        
    Returns
    -------
    xr : float
        x-coordinate in lab frame.
    yr : float
        y-coordinate in lab frame.
    zr : float
        z-coordinate in lab frame.
    """
    x1,y1,z1 = multiplyRbeta(x,y,z,beta)   #  
    xr,yr,zr = multiplyRalpha(x1,y1,z1,alpha)#  
    
    return xr, yr, zr

def render_0kl_reflections_parametric(q0,deltaQ0,wMu,mu,pGamma,alpha=0,beta=0,\
                                      Nt=50,Nphi=50,pcolor=(0,0,1),popacity=1):
    muBar = np.pi/2.0 - mu
    print("Nt= ",Nt, "Nphi= ",Nphi)
    dt, dphi = 2*np.pi/Nt, 2*np.pi/Nphi
    t,phi=np.mgrid[0:2*np.pi+1.5*dt:dt, 0:2*np.pi+1.5*dphi:dphi]
    # qypp, qypm: +/- signs for qy coordinate (above and below qx-qz plane)
    qxp  =  (q0+deltaQ0*pGamma*np.sin(t))*np.cos(muBar+pGamma*wMu*np.cos(t))*np.cos(phi)
    qypp =  (q0+deltaQ0*pGamma*np.sin(t))*np.sin(muBar+pGamma*wMu*np.cos(t))
    qypm = -(q0+deltaQ0*pGamma*np.sin(t))*np.sin(muBar+pGamma*wMu*np.cos(t))
    qzp  =  (q0+deltaQ0*pGamma*np.sin(t))*np.cos(muBar+pGamma*wMu*np.cos(t))*np.sin(phi)
    
    qxpR1,qyppR1,qzpR1 = rotated_vectors(qxp,qypp,qzp,alpha,beta) #-ve sign in function
    qxpR2,qypmR2,qzpR2 = rotated_vectors(qxp,qypm,qzp,alpha,beta) #-ve sign in function

    #mlab.mesh(qxpR1,qyppR1,qzpR1,color=pcolor,opacity=popacity)
    #mlab.mesh(qxpR2,qypmR2,qzpR2,color=pcolor,opacity=popacity)
    
    return 

def render_0kl_reflections_parametric_flat(q0,deltaQ0,wMu,mu,pGamma,aniso=2.0,alpha=0,beta=0,\
                                      Nt=50,Nphi=50,pcolor=(0,0,1),popacity=1):
    muBar = np.pi/2.0 - mu
    dt, dphi = 2*np.pi/Nt, 2*np.pi/Nphi
    t,phi=np.mgrid[0:2*np.pi+1.5*dt:dt, 0:2*np.pi+1.5*dphi:dphi]
    # qypp, qypm: +/- signs for qy coordinate (above and below qx-qz plane)
    qxp  =  (deltaQ0*pGamma*aniso*np.cos(t))*np.cos(phi)
    qypp =  (q0+deltaQ0*pGamma*(1./aniso)*np.sin(t))
    qypm = -(q0+deltaQ0*pGamma*(1./aniso)*np.sin(t))
    qzp  =  (deltaQ0*pGamma*aniso*np.cos(t))*np.sin(phi)
    
    qxpR1,qyppR1,qzpR1 = rotated_vectors(qxp,qypp,qzp,alpha,beta) #-ve sign in function
    qxpR2,qypmR2,qzpR2 = rotated_vectors(qxp,qypm,qzp,alpha,beta) #-ve sign in function

    #mlab.mesh(qxpR1,qyppR1,qzpR1,color=pcolor,opacity=popacity)
    #mlab.mesh(qxpR2,qypmR2,qzpR2,color=pcolor,opacity=popacity)
    
    return 

def render_direct_plane(w,dw,alpha,beta,p_color=(1,0,0),p_opacity=0):
    lx,ly=np.mgrid[-w/2.0:w/2.0:dw, -w/2.0:w/2.0:dw]
    lz = np.zeros_like(lx)
    lxR,lyR,lzR = rotated_vectors(lx,ly,lz,alpha,beta)
    #mlab.mesh(lxR,lyR,lzR,color=p_color,opacity=p_opacity)
    return

def render_reverse_plane(w,dw,alpha,beta,p_color=(1,0,0),p_opacity=0):
    lx,ly=np.mgrid[-w/2.0:w/2.0:dw, -w/2.0:w/2.0:dw]
    lz = np.zeros_like(lx)
    lxR,lyR,lzR = rotated_vectors_rev(lx,ly,lz,alpha,beta)
    #mlab.mesh(lxR,lyR,lzR,color=p_color,opacity=p_opacity)
    return

def intensity_trace_radial_range(c,wavelen,g0,dG0,mu,wMu,q0,dQ,Nfac,aR,bR,\
                                  integrate_bool,wfuncInt,q1,q2,dq=0.01):
    qxt, qyt, qzt = calc_ewald_trace_radial(wavelen,c,q1,q2,dq)
    It = np.zeros_like(qxt)
    if integrate_bool == True:
        It = Iplanarfibril(qxt,qyt,qzt,g0,dG0,mu,q0,wMu,dQ,Nfac,aR,bR,\
                           integrate_bool)/wfuncInt
    else:
        It = Iplanarfibril(qxt,qyt,qzt,g0,dG0,mu,q0,wMu,dQ,Nfac,aR,bR,\
                           integrate_bool)
    return It

def intensity_trace_radial_range_bkp(c,wavelen,g0,dG0,mu,wMu,q0,dQ,Nfac,aR,bR,\
                                  integrate_bool,wfuncInt,q1,q2,dq=0.01):
    qxt, qyt, qzt = calc_ewald_trace_radial(wavelen,c,q1,q2,dq)
    It = np.zeros_like(qxt)
    if integrate_bool == True:
        It = Iplanarfibril(qxt,qyt,qzt,g0,dG0,mu,q0,wMu,dQ,Nfac,aR,bR,\
                           integrate_bool)/wfuncInt
    else:
        It = Iplanarfibril(qxt,qyt,qzt,g0,dG0,mu,q0,wMu,dQ,Nfac,aR,bR,\
                           integrate_bool)
    return It

def intensity_trace_angular_range(q,wavelen,g0,dG0,mu,wMu,q0,dQ,Nfac,aR,bR,\
                                  integrate_bool,wfuncInt,c1,c2,dc=1.0):
    qxt, qyt, qzt = calc_ewald_trace_angular(wavelen,q,c1,c2,dc)
    It = np.zeros_like(qxt)
    if integrate_bool == True:
        It = Iplanarfibril(qxt,qyt,qzt,g0,dG0,mu,q0,wMu,dQ,Nfac,aR,bR,\
                           integrate_bool)/wfuncInt
    else:
        It = Iplanarfibril(qxt,qyt,qzt,g0,dG0,mu,q0,wMu,dQ,Nfac,aR,bR,\
                           integrate_bool)
    return It

def intensity_trace_angular_range_bkp(q,wavelen,g0,dG0,mu,wMu,q0,dQ,Nfac,aR,bR,\
                                  integrate_bool,wfuncInt,c1,c2,dc=1.0):
    qxt, qyt, qzt = calc_ewald_trace_angular(wavelen,q,c1,c2,dc)
    if integrate_bool == True:
        It = Iplanarfibril(qxt,qyt,qzt,g0,dG0,mu,q0,wMu,dQ,Nfac,aR,bR,\
                           integrate_bool)/wfuncInt
    else:
        It = Iplanarfibril(qxt,qyt,qzt,g0,dG0,mu,q0,wMu,dQ,Nfac,aR,bR,\
                           integrate_bool)
    return It

def calc_ewald_trace_singular(wavelen,q,chi):
        
    # Ewald sphere radius
    qE = 2*np.pi/wavelen
    
    qx = q*np.sqrt(1-(q/(2*qE))**2) * np.cos(np.radians(chi))
    qy = q*np.sqrt(1-(q/(2*qE))**2) * np.sin(np.radians(chi))
    qz = np.ones_like(qx)
    qz = (q**2/(2*qE))*qz

    return qx, qy, qz

def calc_ewald_trace_angular(wavelen,q,chi1,chi2,deltachi=1.0):
        
    # Ewald sphere radius
    qE = 2*np.pi/wavelen
    
    chi = np.arange(np.radians(chi1),np.radians(chi2),np.radians(deltachi))
    qx = q*np.sqrt(1-(q/(2*qE))**2) * np.cos(chi)
    qy = q*np.sqrt(1-(q/(2*qE))**2) * np.sin(chi)
    qz = np.ones_like(qx)
    qz = (q**2/(2*qE))*qz

    return qx, qy, qz

def calc_ewald_trace_grid(wavelen,q,chi):
     
    # Ewald sphere radius

    qE = 2*np.pi/wavelen

    qg, chig = np.meshgrid(q,chi)

   

    qx = qg*np.sqrt(1-(qg/(2*qE))**2) * np.cos(np.radians(chig))

    qy = qg*np.sqrt(1-(qg/(2*qE))**2) * np.sin(np.radians(chig))

    qz = np.ones_like(qx)

    qz = (qg**2/(2*qE))*qz

    return qx, qy, qz

def calc_ewald_trace_radial(wavelen,chi,q1,q2,dq=0.01):
        
    # Ewald sphere radius
    qE = 2*np.pi/wavelen
    
    q = np.arange(q1,q2,dq)
    qx = q*np.sqrt(1-(q/(2*qE))**2) * np.cos(np.radians(chi))
    qy = q*np.sqrt(1-(q/(2*qE))**2) * np.sin(np.radians(chi))
    qz = np.ones_like(qx)
    qz = (q**2/(2*qE))

    return qx, qy, qz

def calc_ewald_surface(wavelen,qxDrange,qyDrange,dqD,qx_minus=None,\
                       qy_minus=None,shiftfac=2.1):
    qxp = qxDrange
    qyp = qyDrange
    if qx_minus==None:
        qxm = -qxDrange
    else:
        qxm = qx_minus
    
    qxp = qxDrange
    if qy_minus==None:
        qym = -qyDrange
    else:
        qym = qy_minus   
    
    qxD, qyD = np.mgrid[qxm:qxp:dqD, qym:qyp:dqD]
   
    q = np.sqrt(qxD**2+qyD**2)
    sinChi, cosChi = np.zeros_like(qxD), np.zeros_like(qxD)
    sinChi, cosChi = np.where(q>0,(qyD/q),0), np.where(q>0,(qxD/q),0)
    
    # Ewald sphere radius 
    qE = 2*np.pi/wavelen
    
    qx = q*np.sqrt(1-(q/(2*qE))**2) * cosChi
    qy = q*np.sqrt(1-(q/(2*qE))**2) * sinChi
    qz = (q**2/(2*qE))
    
    qxD_offset = qxD + shiftfac*qxDrange #offset x coordinate
    
    return qx, qy, qz, qxD, qyD, qxD_offset

def fibre_line_coords(L,gamma,NL=2):
    """
    Parameters
    ----------
    L : float
        fibre length.
    gamma : float
        tilt angle of fibre wrt y-axis in x-y plane.
        Like "theta"
    NL : integer, optional
        Number of line points. The default is 2.

    Returns
    -------
    x_rot : float
        x-coord of fibre line.
    y_rot : float
        y-coord of fibre line.
    z_rot : float
        z-coord of fibre line.

    """
    
    Lm, Lp = -L/2.0, L/2.0
    yr = np.linspace(Lm,Lp,NL)
    xr = np.zeros_like(yr)
    zr = np.zeros_like(yr)
    
    x_rot = xr*np.cos(gamma) - yr*np.sin(gamma)
    y_rot = xr*np.sin(gamma) + yr*np.cos(gamma)
    z_rot = zr
    
    return x_rot, y_rot, z_rot

def fibre_full_coords_with_shading(Rf,Lf,gamma,D,w,rho0,rho1,x0=0,y0=0,z0=0,NL=100):
    """
    Parameters
    ----------
    Rf : float
        fibril radius.
    Lf : float
        fibril length.
    gamma : float
        tilt angle in x-y plane, in radians
    D,w,rho0,rho1: float
        electron density shading parameters (fibril_axial_shading)
    Returns
    -------
    xcurv,ycurv,zcurv,xcapt,ycapt,zcapt,xcapb,ycapb,zcapb,density
    """
    xBody,yBody,zBody = fibre_surface_coords(Rf,Lf/2.0,gamma,NL)
    xNoRot,yNoRot,zNoRot = fibre_surface_coords(Rf,Lf/2.0,0,NL)
    x_rot = xBody*np.cos(gamma) + yBody*np.sin(gamma)
    y_rot = -xBody*np.sin(gamma) + yBody*np.cos(gamma)
    z_rot = zBody
    xcurv, ycurv, zcurv = xBody+x0, yBody+y0, zBody+z0
    
    xcp,ycp,zcp = fibre_cap_coords_top(Rf,Lf,gamma)
    xcapt, ycapt, zcapt = xcp+x0, ycp+y0, zcp+z0
    
    xcm,ycm,zcm = fibre_cap_coords_bottom(Rf,Lf,gamma)
    xcapb, ycapb, zcapb = xcm+x0, ycm+y0, zcm+z0
    
    yBodytrunc = np.sqrt(yBody**2+xBody**2) % (2*D)
    yBodytrunc = yNoRot % (2*D)
    density = fibril_axial_shading(yBodytrunc,Lf,D,rho0,rho1,w)
    
    return  xcurv,ycurv,zcurv,xcapt,ycapt,zcapt,xcapb,ycapb,zcapb,density
   

def fibril_axial_shading(y,L,D,rho0,rho1,w):
    """
    Parameters
    ----------
    y : float
        axial coordinate along fibril
    L : float
        fibril length
    D : float
        D-period
    rho0 : float
        constant term in electron density function expression
    rho1 : float
        amplitude term in electron density function expression
    w : float
        broadening term in electron density function

    Returns
    -------
    axial_electron_density(y) = rho0 + rho1*expit((y-x01)/w)*(1-expit((y-x02)/w)))
    x01, x02 = D-D/2.0, D+D/2.0
    """
    x01, x02 = D-D/2.0, D+D/2.0
    edens = np.zeros_like(y)
    edens = rho0 + rho1*(expit((y-x01)/w)*(1-expit((y-x02)/w)))
    return edens

def fibre_cap_coords_top(R,L,gamma,NPhi=100,NR=100):
    dphi = 2*np.pi/NPhi
    dR = R/NR
    
    [r,theta] = np.mgrid[0:R:dR,0:2*np.pi+dphi*1.5:dphi]
    x_body = r*np.cos(theta)
    y_body = (L/2.0)*np.ones_like(x_body)
    z_body = r*np.sin(theta)
    
    x_rot = x_body*np.cos(gamma) + y_body*np.sin(gamma)
    y_rot = x_body*np.sin(gamma) - y_body*np.cos(gamma)
    z_rot = z_body
    
    return x_rot, y_rot, z_rot

def fibre_cap_coords_bottom(R,L,gamma,NPhi=100,NR=100):
    dphi = 2*np.pi/NPhi
    dR = R/NR
    
    [r,theta] = np.mgrid[0:R:dR,0:2*np.pi+dphi*1.5:dphi]
    x_body = r*np.cos(theta)
    y_body = -(L/2.0)*np.ones_like(x_body)
    z_body = r*np.sin(theta)
    
    x_rot = x_body*np.cos(gamma) + y_body*np.sin(gamma)
    y_rot = x_body*np.sin(gamma) - y_body*np.cos(gamma)
    z_rot = z_body
    
    return x_rot, y_rot, z_rot

def fibre_surface_coords(R,L,gamma,NL=100,NPhi=100):
    """
    Returns coordinates of the surface of a fibril with 
    radius R, length L; tilted at angle gamma to y axis in x-y plane and 
    then by angle phi around y axis. Assumes the fibril will
    be part of a Bouligand planar array in x-y, so that the
    tilt around z axis will be accounted for by changing gamma0
    Parameters
    ----------
    R : float
        fibril radius.
    L : float
        fibril length.
    gamma : float
        tilt angle of fibre wrt y-axis in x-y plane.
        Like "theta"
    NL : int, optional
        DESCRIPTION. number of grid points in L; The default is 100.
    NPhi : int, optional
        DESCRIPTION. number of grid points in phi; The default is 100.

    Returns
    -------
    (x,y,z) coordinates in body fixed frame
    x_body : float
        x coordinates of fibril surface.
    y_body : TYPE
        y coordinates of fibril surface.
    z_body : TYPE
        z coordinates of fibril surface.

    """
    dL = L/(NL*1.0)
    dphi = 2*np.pi/NPhi
    [phi, z] = np.mgrid[0:2*np.pi + dphi * 1.5:dphi, -L:L:dL]
    
    x_body = R*np.cos(phi)
    y_body = z
    z_body = R*np.sin(phi)
    
    x_rot = x_body*np.cos(gamma) + y_body*np.sin(gamma)
    y_rot = x_body*np.sin(gamma) - y_body*np.cos(gamma)
    z_rot = z_body
            
    return x_rot, y_rot, z_rot

if __name__ == '__main__':
    # booleans on what to plot
    show_flat_plane = False
    show_tilted_plane = False
    show_reverse_plane = False
    show_fibre_dist = True
    show_ewald_surface = True
    show_2D_pattern = True
    show_parametric_surface = True
    show_main_fibre = True
    #Scattering function parameters
    """
    Typical parameter values (all in nm-1)
    SAXS channel
    q0 = 0.5, qxDrange, qyDrange, dqD = 1.0, 1.0, 0.01, deltaQ0 = 0.01,
    wMu = 0.2
    L = 1.0, fibril radius R = 0.05
    w, dw, opacity = 2.0, 0.02, 0.3
    pGamma=2.0
    
    WAXD channel
    q0 = 15.0, qxDrange, qyDrange, dqD = 22.0, 22.0, 0.2, deltaQ0 = 0.5
    L = 15.0, fibril radius R = 0.3
    w, dw, opacity = 40.0, 2.0, 0.3
    pGamma = 0.5
    
    VVVVVVVVVVVVVVVVVVVVV
    INITIALISE PARAMETERS
    VVVVVVVVVVVVVVVVVVVVV
    """
    # flat scattering; flat = 1 -> flat ellipsoids; flat = 0 -> spherical sectors
    # default flat = 0
    flat=0
    # fibril length 2L, fibril radius R
    #L, R = 15.0, 0.3 #WAXD
    L, R = 1.0, 0.02 #SAXS
    wavelen = 0.07293 # 17 keV
    gamma0 = np.radians(0.0)
    deltaGamma0 = .01
    NL,NPhi = 50,50
    Nt,Nphi=50,50
    alphaR,betaR = np.radians(30),np.radians(0.0)   
    #alphaR,betaR = np.radians(75),np.radians(0.0)   
    # WAXD PARAMETERS VVVVVV
    """
    #WAXD:
    q0, deltaQ0 = 15.0, 0.5 
    wMu = 0.3
    wMu = 0.1
    mu = np.radians(30.0)  # wrt y-axis
    muBar = np.pi/2.0 - mu # wrt x-axis
    pGamma=2.0
    w, dw, opacity = 40.0, 2.0, 0.3
    """
    
    #WAXD
    #qxDrange, qyDrange, dqD = 22.0, 22.0, 0.3
    #SAXS
    qxDrange, qyDrange, dqD = 1.0, 1.0, 0.01
    #qxDrange, qyDrange, dqD = 1.0, 1.0, 0.2
    
    #"""
    #SAXS PARAMETERS VVVVVV
    q0, deltaQ0 = 0.50, 0.01  
    wMu = 0.2 # collagen 
    #wMu = 0.05 # equatorial 
    # collagen: 0.2, wide hat; chitin: 0.02, sharp; in between: 0.05, wide belt
    #"""
    mu = np.radians(0.0)  
    muBar = np.pi/2.0 - mu 
    aniso = 3.0
    pGamma=3.0
    flat=1
    """
    flat
    pGamma=30.0 
    pGamma=10.0 
    """
    w, dw, opacity = 2.0, 0.02, 0.3
    #""" #PARAMETER EXPLANATION IN QCOMMENTS vvvv
    """
    q0: # e.g. for collagen, D = 5 * 2*pi/q0
    deltaQ0: # like wq in fit function
    wMu: # like wchi, but note units in radians, not degrees
    mu: # wrt y-axis
    muBar: # wrt x-axis
    pGamma: #defines the isointensity contour for parametric plot
    
    ^^^^^^^^^^^^^^^^^^^^^
    DONE: INITIALISE PARAMETERS
    ^^^^^^^^^^^^^^^^^^^^^    
    open and initialise Mayavi window instance
    """        
    mlab.clf()

    """
    render real space fibrils
    """    
    psi, z = np.mgrid[0:2*np.pi:2*np.pi/50.0, -L:L:L/50.0]
    xc, yc, zc = R*np.cos(psi), z, R*np.sin(psi)
    mlab.mesh(xc,yc,zc,color=(0,1,0))
    
    xBody,yBody,zBody = fibre_surface_coords(R,L,gamma0,NL,NPhi)
    xR,yR,zR = rotated_vectors(xBody,yBody,zBody,alphaR,betaR) #-ve sign in function
    mlab.mesh(xR,yR,zR,color=(1,1,0))
    
    mlab.orientation_axes()
    mlab.view(20,30,roll=0)
    #mlab.figure(bgcolor=(1,1,1))
    

    #qxDrange, qyDrange, dqD = 1.0, 1.0, 0.011
    #qxDrange, qyDrange, dqD = 1.0, 1.0, 0.01
    
    if show_parametric_surface == True:
        
        pcolor=(0,0,1)
        popacity=0.2
        """
        def render_0kl_reflections_parametric(q0,deltaQ0,wMu,mu,pGamma,alpha=0,beta=0,\
                                      Nt=50,Nphi=50,pcolor=(0,0,1),popacity=1):
        """
        if flat==0:
            """
            def render_0kl_reflections_parametric(q0,deltaQ0,wMu,mu,pGamma,alpha=0,beta=0,\
                                      Nt=50,Nphi=50,pcolor=(0,0,1),popacity=1):
            """
            render_0kl_reflections_parametric(q0,deltaQ0,wMu,mu,pGamma,0,0,\
                                          Nt=Nt,Nphi=Nphi,pcolor=pcolor,popacity=popacity)
        else:
            render_0kl_reflections_parametric_flat(q0,deltaQ0,wMu,mu,pGamma,aniso,0,0,\
                                          Nt=Nt,Nphi=Nphi,pcolor=pcolor,popacity=popacity)
                
        pcolor=(0,1,1)
        popacity=0.7
        if flat==0:
            render_0kl_reflections_parametric(q0,deltaQ0,wMu,mu,pGamma,\
                                          alphaR,betaR,Nt=Nt,Nphi=Nphi,pcolor=pcolor,popacity=popacity)
        else:
            render_0kl_reflections_parametric_flat(q0,deltaQ0,wMu,mu,pGamma,aniso,\
                                          alphaR,betaR,Nt=Nt,Nphi=Nphi,pcolor=pcolor,popacity=popacity)
     
    """
    add plane: perpendicular to beam, tilted by fibre tilt and
           tilted in reverse way to see what the detector
           plane sees
    """
    color_flat, color_direct_tilt, color_reverse_tilt = (1,0,0),(0,1,1),(0,1,1)
    
    if show_flat_plane == True:
        render_direct_plane(w,dw,0,0,color_flat,0.7)
    
    if show_tilted_plane == True:
        render_direct_plane(w,dw,alphaR,betaR,color_direct_tilt,opacity)
        
    if show_reverse_plane == True:
        render_reverse_plane(w,dw,alphaR,betaR,color_reverse_tilt,opacity)
       
    """
    define the curved surface (qx, qy, qz) for the detector intersection
    Ewald sphere intersection
    """
    qx, qy, qz, qxD, qyD, qxD_offset = \
        calc_ewald_surface(wavelen,qxDrange,qyDrange,dqD)
    
    if show_ewald_surface == True:
        mlab.mesh(qx,qy,qz,opacity=0.3)
    
    
    """
    Generate I(qx,qy,qz) set for a selected (qx,qy,qz) tuple
    This is generalisable for a 2D surface (as above (qx,qy,qz)) 
    or a 1D trace (not shown here, called in other scripts using this module)
    
    'integrate' is a Boolean. When True it tells the function to apply the 
    weighting function via the convolution integral. When False, the integral
    is skipped and the kernel is evaluated at gamma = gamma0. This is faster.
    
    This is used for a) doing the SAXS-based calculations where the widening
    of wMu replaces the wgamma as a way to show conical fibril orientation and 
    b) the WAXD case where we want to show the diffraction from a single fibril
    e.g. if the beam is small enough that it goes through a single sublamella
    in cuticle.
    """
    normalising_Factor = normFac(mu,wMu,deltaQ0)
    integrate=False
    if integrate == True:
        """
        calculate integral of weighting function to normalise
        """ 
        wgammaInt = quad(wgamma,-np.pi/2.0,np.pi/2.0,args=(gamma0,deltaGamma0), \
                         epsrel=1.0e-3)[0]
    else:
        wgammaInt = 1.0
    
    print("starting")
    t1 = time()
    print("flat: ", flat)
    delta = 1.00
    Ichi_013 = Iplanarfibril(qx,qy,qz,gamma0,deltaGamma0,mu,q0,wMu,deltaQ0,normalising_Factor,\
                  alphaR,betaR,integrate,flat=flat,delta=delta)/wgammaInt
    t2 = time()
    print("ending: ",t2-t1)
    
    """
    show 2D pattern on detector
    set max and min intensity levels for contour plot
    be careful that the function appears to have singularities at qy=0
    so maximum intensity at these values will be very high and the physical
    peak will not be visible
    """
    minInt = 0.0
    #maxInt = 1./(2*np.pi*wMu*deltaQ0)    
    maxInt = np.max(Ichi_013)
    #maxInt = 2125.0
    qzDO = np.zeros_like(qxD_offset)
    
    if show_2D_pattern == True:
        mlab.mesh(qxD_offset,qyD,qzDO,scalars=Ichi_013,vmax=maxInt,vmin=minInt)
        
    """
    pseudocode for rendering fibrils in different directions
    
    select a range of angles (theta, phi) at which to represent the fibre
    orientations (meshgrid)
    
    generate a set of radii and lengths weighted by PDF at this orientations
    (same shape, meshgrid)
    
    R = Rmin + w(phi, theta)*(Rmax-Rmin)
    
    for (theta, phi, l, r) in zip(theta, phi, l, r)
        gamma0 = 0
        xf, yf, zf = fibre_surface_coords(r,l,gamma0,NL,NPhi)
        xR,yR,zR = rotated_vectors(xf,yf,zf,theta,phi) #-ve sign in function
        mlab.mesh(xR,yR,zR,color=(1,1,0))
    """
    dthetaF, dphiF  = np.pi / 10.0, np.pi / 10.0
    [thetaF, phiF] = np.mgrid[0:np.pi - dthetaF * 0.01:dthetaF,\
                              0:2 * np.pi - dphiF * 0.01:dphiF]
    Lfmin, Lfmax = 0.2*L,1.0*L
    Rfmin, Rfmax = 0.4*R,1.0*R
    """
    def w3D(theta,A,B)
    """
    A,B=1.0,1.0
    w3DInt = quad(w3D,0,np.pi/2.0, args=(A,B), epsrel=1.0e-3)[0]
    
    Lf = Lfmin + (Lfmax-Lfmin)*(w3D(thetaF,A,B)/w3DInt)
    Rf = Rfmin + (Rfmax-Rfmin)*(w3D(thetaF,A,B)/w3DInt)
    
    thetaFF, phiFF, LfF, RfF = thetaF.flatten(), phiF.flatten(), Lf.flatten(),\
        Rf.flatten()
   
    fcolor, fintcolor = (0,1,0), (0,0,1)
    fopacity, fintopacity = 0.025, 1.0    
    
    show_intersecting_fibres = True # plots fibres intersecting plane in diff color
    deltaPhi = np.radians(5.0)      # azimuthal angle diff from detector plane
    deltaZ = (Lfmin+Lfmax)/20.0      # azimuthal angle diff from detector plane
    
    if show_fibre_dist==True:
        for (theta,phi,l,r) in zip(thetaFF,phiFF,LfF,RfF):
            #print(theta,phi,l,r)
            gamma0 = 0.0
            #xf, yf, zf = vfsc(r,l,gamma0,NL,NPhi)
            xf, yf, zf = fibre_surface_coords(r,l,gamma0,NL,NPhi)
            xtp,ytp,ztp = rotated_vectors(xf,yf,zf,theta,phi) 
            xR,yR,zR = rotated_vectors(xtp,ytp,ztp,alphaR,betaR)
            
            if show_intersecting_fibres == True:
#                if (np.fabs(phi-np.pi/2.0)<deltaPhi):
                if (np.max(np.fabs(zR))<deltaZ):
                    mlab.mesh(xR,yR,zR,color=fintcolor, opacity=fintopacity)
                else:
                    mlab.mesh(xR,yR,zR,color=fcolor, opacity=fopacity)
            else:
                mlab.mesh(xR,yR,zR,color=fcolor, opacity=fopacity)
    mlab.show(stop=True)
