import numpy as np
from numpy import array

from Parameters import h,N,X,Y,Omega, D1,D2,rho_10,rho_20, V_1,V_2,DV_1,DV_2, alpha

def D_X(V,h):
    direction = -int(np.sign(h))
    return (np.roll(V,direction,axis=1) -np.roll(V,-direction,axis=1) )/(2*h)

def D_Y(V,h):
    direction = -int(np.sign(h))
    return (np.roll(V,direction,axis=0) -np.roll(V,-direction,axis=0))/(2*h)
def D(V,h):
    return array([D_X(V,h),D_Y(V,h)])
def div(V,h):
    return D_X(V[0],h) + D_Y(V[1],h)

def evolve(dt,EndTime,minChange,method = 'forward_euler'):
    data = {'method' : method, 'D1,D2' : (D1,D2), 'minChange' : minChange, 'dt': dt,  'rho_10,rho_20' : (rho_10,rho_20)}
    TimeSteps = int(EndTime/dt)
    rho_1 = rho_10
    rho_2 = rho_20
    t = 0
    MaxSize = 1000
    if TimeSteps < MaxSize:
        scale = 1
        rho_t_1 = np.zeros((1+TimeSteps,N,N))
        rho_t_2 = np.zeros((1+TimeSteps,N,N))
    else:
        scale = round(TimeSteps/MaxSize)
        rho_t_1 = np.zeros((MaxSize+1,N,N))
        rho_t_2 = np.zeros((MaxSize+1,N,N))
    i = 0
    rho_t_1[i] = rho_1
    rho_t_2[i] = rho_2
    LastUpdate= 0
    while i< TimeSteps:
        if method == 'forward_euler':
            rho_1,rho_2,Magnitude_change =  forward_euler(rho_1,rho_2,dt)
        elif method == 'crank_nicholson':
            rho_1,rho_2,Magnitude_change =  crank_nicholson(rho_1,rho_2,dt)
        elif method == 'backwards_euler':
            rho_1,rho_2,Magnitude_change =  backwards_euler(rho_1,rho_2,dt)

        t +=dt
        i += 1
        if int((100*i/TimeSteps)%1) ==0 and int(100*i/TimeSteps) >0 and int((100*i/TimeSteps))>LastUpdate: 
            print(str(int((100*i/TimeSteps))) + "% completed. Current value of dp/dt = "+ str(Magnitude_change))
            LastUpdate = int((100*i/TimeSteps))
        if i%scale == 0:
            rho_t_1[int(i/scale)] = rho_1
            rho_t_2[int(i/scale)] = rho_2
        if len(np.argwhere(rho_1 <0 )) >0 or  len(np.argwhere(rho_2 <0 )) >0 or  len(np.argwhere(rho_1 +rho_2>1 )) >0 or Magnitude_change < minChange:
            print('ended early at  i='+str(i))
            print('np.argwhere(rho_1 <0 ) = ' + str(np.argwhere(rho_1 <0 )))
            print('np.argwhere(rho_1 +rho_2>1 ) = ' + str(np.argwhere(rho_1 +rho_2>1 )))
            print('np.argwhere(rho_2 <0 ) = ' + str(np.argwhere(rho_2 <0 )))
            print('Magnitude_change = ' +str(Magnitude_change))
            data['rho_t_1'] = rho_t_1
            data['rho_t_2'] = rho_t_2
            data['t'] = np.linspace(0,dt*(i+1),int((i+1)/scale))
            return rho_t_1[:int((i+1)/scale),:,:],rho_t_2[:int((i+1)/scale),:,:],np.linspace(0,dt*(i+1),int((i+1)/scale)), data
            break
        #print('Got to i = ' +str(i)+', successfully!')
    data['rho_t_1'] = rho_t_1
    data['rho_t_2'] = rho_t_2
    data['t'] = np.linspace(0,dt*(i+1),int((i+1)/scale))
    return rho_t_1,rho_t_2,np.linspace(0,dt*(i+1),int((i+1)/scale)), data
def forward_euler(rho_1,rho_2,dt):
    delta_rho = dp(rho_1,rho_2)
    rhos_new = array([rho_1,rho_2])
    rhos_new +=  dt*delta_rho
    Magnitude_change = np.sqrt(np.sum(delta_rho**2)*(h**2))
    rhos_new = array([rho_1,rho_2])
    rhos_new +=  dt*delta_rho
    Magnitude_change = np.sqrt(np.sum(delta_rho**2)*(h**2))
    #print(np.sqrt(np.amax( (J_plus_X - J_minus_X + J_plus_Y - J_minus_Y)**2)))
    return rhos_new[0],rhos_new[1],Magnitude_change

def crank_nicholson(rho_1,rho_2,dt):
    rhos_0 = array([rho_1,rho_2])
    dp0 = dp(rhos_0[0],rhos_0[1])
    phi = lambda rhos: rhos_0 + (dt/2)*(dp0 + dp(rhos[0],rhos[1]))
    rhos = rhos_0 + dt*dp0
    for k in range(0,20):
        new_rhos = phi(rhos)
        if np.linalg.norm(new_rhos - rhos)*h < 1e-8:
           # print("Crank-Nicholson converged after "  +str(k) + " Error  = " + str(np.linalg.norm(new_rhos - rhos)))
            break
        rhos = new_rhos
    delta_rho = (rhos - rhos_0)/dt
    Magnitude_change = np.sqrt(np.sum(delta_rho**2)*(h**2))
    #print(np.sqrt(np.amax( (J_plus_X - J_minus_X + J_plus_Y - J_minus_Y)**2)))
    return rhos[0],rhos[1],Magnitude_change

def backwards_euler(rho_1,rho_2,dt):
    rhos_0 = array([rho_1,rho_2])
    dp0 = dp(rhos_0[0],rhos_0[1])
    phi = lambda rhos: rhos_0 + (dt)*(dp(rhos[0],rhos[1]))
    rhos = rhos_0 + dt*dp0
    for k in range(0,20):
        new_rhos = phi(rhos)
        if np.linalg.norm(new_rhos - rhos)*h < 1e-12:
           # print("Crank-Nicholson converged after "  +str(k) + " Error  = " + str(np.linalg.norm(new_rhos - rhos)))
            break
        rhos = new_rhos
    delta_rho = (rhos - rhos_0)/dt
    Magnitude_change = np.sqrt(np.sum(delta_rho**2)*(h**2))
    #print(np.sqrt(np.amax( (J_plus_X - J_minus_X + J_plus_Y - J_minus_Y)**2)))
    return rhos[0],rhos[1],Magnitude_change


def dp(rho_1,rho_2):   # value of \partial_t (\rho_1,\rho_2)
    g1 = 2*D1/(D1 +D2)
    g2 = 2*D2/(D1 +D2)
    rho = rho_1 +rho_2
    Drho_1 = D(rho_1,h)
    Drho_2 = D(rho_2,h)
    Drho = D(rho,h)
    DV1 = DV_1(X,Y)
    DV2 = DV_2(X,Y)
    C = (D1+D2)*g1*g2*alpha/2
    W_1 = D1*(1-rho)*(1-g1*alpha*rho_1)*Drho_1 + D1*rho_1*(1-g1*alpha*rho)*Drho + D1*(1-rho)*rho_1*(1-g1*alpha*rho_1)*DV1 +C*(rho_1*(1-rho)*Drho_2 + rho_1*rho_2*Drho+rho_1*rho_2*(1-rho)*DV2)
    W_2 = D2*(1-rho)*(1-g2*alpha*rho_2)*Drho_2 + D2*rho_2*(1-g2*alpha*rho)*Drho + D2*(1-rho)*rho_2*(1-g2*alpha*rho_2)*DV2 +C*(rho_2*(1-rho)*Drho_1 + rho_2*rho_1*Drho+rho_2*rho_1*(1-rho)*DV1)
    J_1 = div(W_1,h)
    J_2 = div(W_2,h)
    return array([J_1,J_2])

def M(rho_1,rho_2):
    g1 = 2*D1/(D1 +D2)
    g2 = 2*D2/(D1 +D2)
    rho = rho_1 + rho_2
    M = np.array([[D1*rho_1*(1-rho)*(1-g1*alpha*rho_2),(D1+D2)*g1*g2*alpha*(1-rho)*rho_1*rho_2/2],[(D1+D2)*g1*g2*alpha*(1-rho)*rho_1*rho_2/2,D2*rho_2*(1-rho)*(1-g2*alpha*rho_1)]])
    return M
    

def dw(rho_1,rho_2):   # value of \partial_t (\rho_1,\rho_2)
    g1 = 2*D1/(D1 +D2)
    g2 = 2*D2/(D1 +D2)
    rho = rho_1 +rho_2
    Drho_1 = D(rho_1,h)
    Drho_2 = D(rho_2,h)
    Drho = D(rho,h)
    DV1 = DV_1(X,Y)
    DV2 = DV_2(X,Y)
    return array([ Drho_1/rho + Drho/(1-rho) + DV1,Drho_2/rho + Drho/(1-rho) + DV2 ])
def dE(rho_1,rho_2):
    v=dw(rho_1,rho_2)
    Mob = M(rho_1,rho_2)
    return np.einsum( 'ijkl,ijkl' ,v,(np.einsum( 'itkl,tjkl->ijkl', Mob,v)))*(h**2)

def E(rho_1,rho_2):
    I1 = np.nonzero(rho_1)
    I2 = np.nonzero(rho_2)
    rho = rho_1 +rho_2
    I3 = np.nonzero(1-rho)
    E_0 = h**2*(np.sum(rho_1[I1]*np.log(rho_1[I1])) +np.sum(rho_2[I2]*np.log(rho_2[I2]))  +np.sum((1- rho[I3])*np.log(1-rho[I3])))
    E_V = h**2*np.sum(rho_1*V_1(X,Y) + rho_2*V_2(X,Y))
    return E_0 +E_V

def alt_evolve(dt,EndTime,minChange,method = 'forward_euler'):
    data = {'method' : method, 'D1,D2' : (D1,D2), 'minChange' : minChange, 'dt': dt,  'rho_10,rho_20' : (rho_10,rho_20)}
    TimeSteps = int(EndTime/dt)
    rho_1 = rho_10
    rho_2 = rho_20
    t = 0
    MaxSize = 1000
    if TimeSteps < MaxSize:
        scale = 1
        rho_t_1 = np.zeros((1+TimeSteps,N,N))
        rho_t_2 = np.zeros((1+TimeSteps,N,N))
    else:
        scale = round(TimeSteps/MaxSize)
        rho_t_1 = np.zeros((MaxSize+1,N,N))
        rho_t_2 = np.zeros((MaxSize+1,N,N))
    i = 0
    rho_t_1[i] = rho_1
    rho_t_2[i] = rho_2
    LastUpdate= 0
    while i< TimeSteps:
        if method == 'forward_euler':
            rho_1,rho_2,Magnitude_change =  alt_forward_euler(rho_1,rho_2,dt)
        elif method == 'crank_nicholson':
            rho_1,rho_2,Magnitude_change =  crank_nicholson(rho_1,rho_2,dt)
        elif method == 'backwards_euler':
            rho_1,rho_2,Magnitude_change =  backwards_euler(rho_1,rho_2,dt)

        t +=dt
        i += 1
        if int((100*i/TimeSteps)%5) ==0 and int(100*i/TimeSteps) >0 and int((100*i/TimeSteps))>LastUpdate: 
            print(str(int((100*i/TimeSteps))) + "% completed. Current value L2 norm of dp/dt = "+ str(Magnitude_change))
            LastUpdate = int((100*i/TimeSteps))
        if i%scale == 0:
            rho_t_1[int(i/scale)] = rho_1
            rho_t_2[int(i/scale)] = rho_2
        if len(np.argwhere(rho_1 <0 )) >0 or  len(np.argwhere(rho_2 <0 )) >0 or  len(np.argwhere(rho_1 +rho_2>1 )) >0 or Magnitude_change < minChange:
            print('ended early at  i='+str(i))
            print('np.argwhere(rho_1 <0 ) = ' + str(np.argwhere(rho_1 <0 )))
            print('np.argwhere(rho_1 +rho_2>1 ) = ' + str(np.argwhere(rho_1 +rho_2>1 )))
            print('np.argwhere(rho_ <0 ) = ' + str(np.argwhere(rho_2 <0 )))
            print('Magnitude_change = ' +str(Magnitude_change))
            data['rho_t_1'] = rho_t_1
            data['rho_t_2'] = rho_t_2
            data['t'] = np.linspace(0,dt*(i+1),int((i+1)/scale))
            return rho_t_1[:int((i+1)/scale),:,:],rho_t_2[:int((i+1)/scale),:,:],np.linspace(0,dt*(i+1),int((i+1)/scale)), data
            break
        #print('Got to i = ' +str(i)+', successfully!')
    data['rho_t_1'] = rho_t_1
    data['rho_t_2'] = rho_t_2
    data['t'] = np.linspace(0,dt*(i+1),int((i+1)/scale))
    return rho_t_1,rho_t_2,np.linspace(0,dt*(i+1),int((i+1)/scale)), data
def alt_forward_euler(rho_1,rho_2,dt):
    v=dw(rho_1,rho_2)
    Mob = M(rho_1,rho_2)
    F = np.einsum( 'itkl,tjkl->ijkl', Mob,v)
    J1 = div(F[0,:,:],h)
    J2 = div(F[1,:,:],h)
    delta_rho = array([J1,J2])
    rhos_new = array([rho_1,rho_2])
    rhos_new +=  dt*delta_rho
    Magnitude_change = np.sqrt(np.sum(delta_rho**2)*(h**2))
    #print(np.sqrt(np.amax( (J_plus_X - J_minus_X + J_plus_Y - J_minus_Y)**2)))
    return rhos_new[0],rhos_new[1],Magnitude_change