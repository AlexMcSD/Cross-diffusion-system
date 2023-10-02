import numpy as np
from numpy import array

def makeSystem(N, dim = 2):
    h = 1/N
    Omega = np.meshgrid(np.linspace(0,1-h,N),np.linspace(0,1-h,N))
    return Omega, h,N
# set paramters of lattice
#Omega, h,N = makeSystem(1000, dim = 2)

#savefile = 'Stable (constant) initial conditions'
#savefile = 'Stable initial conditions'
#savefile = 'Unstable initial'
savefile = 'Stable initial conditions'

Omega, h,N = makeSystem(100, dim = 2)
X,Y =Omega

alpha = np.pi/2 - 1
if savefile == 'Stable (constant) initial conditions':
    V_1 = lambda x,y : np.sin(2*np.pi*x)
    V_2 = lambda x,y :  -np.sin(2*np.pi*x) # 0*np.cos(2*np.pi*x)
    DV_1 = lambda x,y: array( [2*np.pi*np.cos( 2*np.pi*x), 0*x])
    DV_2 = lambda x,y: -array( [2*np.pi*np.cos( 2*np.pi*x), 0*x]) 
    D1 = 1
    D2 = 1
    rho_10 = 0*X + .4
    rho_20 = 0*X +.4
elif savefile == 'Stable initial conditions':
    V_1 = lambda x,y : np.sin(2*np.pi*x)
    V_2 = lambda x,y :  -np.sin(2*np.pi*x) # 0*np.cos(2*np.pi*x)
    DV_1 = lambda x,y: array( [2*np.pi*np.cos( 2*np.pi*x), 0*x])
    DV_2 = lambda x,y: -array( [2*np.pi*np.cos( 2*np.pi*x), 0*x]) 
    D1 = 1
    D2 = 1
    rho_10 = 0*X + .01
    rho_20 = 0*X +.01
    I = np.argwhere(X[0,:]  < 0.5)
    rho_10[:,I] = 0.4
    J = np.argwhere(X[0,:]  > 0.5)
    rho_20[:,J] = 0.4
elif savefile == 'Unstable initial':
    V_1 = lambda x,y : 0*np.sin(2*np.pi*x)
    V_2 = lambda x,y :  -np.sin(2*np.pi*x) # 0*np.cos(2*np.pi*x)
    DV_1 = lambda x,y: 0*array( [2*np.pi*np.cos( 2*np.pi*x), 0*x])
    DV_2 = lambda x,y: -array( [2*np.pi*np.cos( 2*np.pi*x), 0*x]) 
    D1 = 1
    D2 = 20
    rho_10 = 0*X + .93
    rho_20 = 0*X +.06
elif savefile == 'Unstable initial refined':
    V_1 = lambda x,y : 0*np.sin(2*np.pi*x)
    V_2 = lambda x,y :  -np.sin(2*np.pi*x) # 0*np.cos(2*np.pi*x)
    DV_1 = lambda x,y: 0*array( [2*np.pi*np.cos( 2*np.pi*x), 0*x])
    DV_2 = lambda x,y: -array( [2*np.pi*np.cos( 2*np.pi*x), 0*x]) 
    D1 = 1
    D2 = 20
    Omega, h,N = makeSystem(1000, dim = 2)
    X,Y =Omega
    rho_10 = 0*X + .93
    rho_20 = 0*X +.06
