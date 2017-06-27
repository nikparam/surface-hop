# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 22:17:17 2017

@author: User
"""
import math
import matplotlib.pyplot as plt
import time

############################################################
#Parameters of system
############################################################

A = [0.01,0.1,0.0006]
B = [1.6,0.28,0.1]
C = [0.005,0.015,0.9]
D = [1.,0.06]
E_0 = 0.05
shift = 3
k=0.81
m=2000

k0 = 0
x0 = -2 
N = -4

omega = math.sqrt(k/m)
alpha = m*omega

sigma = math.sqrt(2/m/omega)
sigma_i = 1/sigma
Norm = math.sqrt(1/math.sqrt(2*math.pi*sigma))

############################################################
#Potential and Kinetic energies
############################################################

def _potential(x,N):
    V = [0,0,0]
    if N in [-1,1]:
        if x >= 0:  V[0] = A[0]*(1-math.exp(-B[0]*x))
        else : V[0] = -A[0]*(1-math.exp(B[0]*x))
        V[1] = -V[0]
        V[2] = C[0]*math.exp(-D[0]*x**2)
    if N in [-2,2]:
        V[0] = 0
        V[1] = -A[1]*math.exp(-B[1]*x**2)+E_0
        V[2] = C[1]*math.exp(-D[1]*x**2)
    if N in [-3,3]:
        V[0] = -A[2]
        V[1] = A[2]
        if x < 0: V[2] = B[2]*math.exp(C[2]*x)
        else: V[2] = B[2]*(2-math.exp(-C[2]*x))
    if N in [-4,4]:
        V[0] = k*x*x*0.5
        V[1] = k*(x-shift)*(x-shift)*0.5-omega
        V[2] = omega*0.5
    return V
   
def _diag(M):
    eigVal = ['','']
    eigVal[0] = 0.5*((M[0]+M[1])-math.sqrt((M[0]-M[1])*(M[0]-M[1])+4*M[2]*M[2]))
    eigVal[1] = 0.5*((M[0]+M[1])+math.sqrt((M[0]-M[1])*(M[0]-M[1])+4*M[2]*M[2]))

    eigVec = ['','']
    D = math.sqrt((M[1]-M[0])*(M[1]-M[0])+4*M[2]*M[2])
    cos_phi = math.sqrt(0.5*(1+abs(M[1]-M[0])/D))
    sin_phi = math.sqrt(0.5*(1-abs(M[1]-M[0])/D))

    eigVec[0] = cos_phi
    eigVec[1] = sin_phi

    return eigVal,eigVec

def _kinetic(p):
    return 0.5*p*p/m

############################################################
#Wave function
############################################################

def _Psi(x,i):
    if bin(i)[-1] == '0':
        return Norm*math.exp(-0.25*sigma_i*(x-x0)*(x-x0))*math.cos(k0*(x-x0))
    elif bin(i)[-1] == '1':
        return Norm*math.exp(-0.25*sigma_i*(x-x0)*(x-x0))*math.sin(k0*(x-x0))

############################################################
#Propagating of WF
############################################################

def _Propagate(Re_Psi,Im_Psi,dt,prop,i):
    if bin(i)[-1] == '0':
        x = Re_Psi*math.cos(dt*prop)+Im_Psi*math.sin(dt*prop)
    if bin(i)[-1] == '1':
        x = -Re_Psi*math.sin(dt*prop)+Im_Psi*math.cos(dt*prop)
    return x

############################################################
#Fast Fourier transform
############################################################

def _SWAP(r,l):
    r,l = l,r
    return r,l

def _FFT(data,isign,nn,delta):
    n = nn << 1
    j = 0
    for i in range(0,n-1,2):
        if j>i:
            data[i],data[j] = _SWAP(data[i],data[j])
            data[i+1],data[j+1] = _SWAP(data[i+1],data[j+1])
        m = nn
        while (m >= 2 and j > m-1):
            j -= m
            m >>= 1
        j+= m

    mmax = 2
    while n > mmax:
        istep = mmax << 1
        theta = isign*2.0*math.pi/mmax
        wtemp = math.sin(0.5*theta)
        wpr = -2.0*wtemp*wtemp
        wpi = math.sin(theta)
        wr = 1.0
        wi = 0.0
        for m in range(0,mmax-1,2):
            for i in range(m,n,istep):
                j = i+mmax
                tempr = wr*data[j]-wi*data[j+1]
                tempi = wr*data[j+1]+wi*data[j]
                data[j] = data[i]-tempr
                data[j+1] = data[i+1]-tempi
                data[i] += tempr
                data[i+1] += tempi
            wtemp = wr
            wr = wtemp*wpr-wi*wpi+wr
            wi = wi*wpr+wtemp*wpi+wi
        mmax = istep
        
    return data

############################################################
#FFT normalization
############################################################

def _FFT_norm(Psi,nn):
    Norm = 1/nn
    for i in range(nn << 1):
        Psi[i] *= Norm
    return Psi

############################################################
#Integration
############################################################
    
def _integration(N,delta,array_y):
    integral=0.0    
    for i in range(N):
        if i == 0:
            integral += (array_y[0]+array_y[1])*delta*0.5
        elif i < N-2 and i > 0:
            integral += (array_y[i+1]+array_y[i-1]+4*array_y[i])*delta/6
        elif i == (N-2):
            integral += (array_y[i]+array_y[i+1])*delta*0.5 
    return integral

############################################################
#Main function starts here -->
############################################################

start_time = time.time()

############################################################
#Time step, number of steps and max time
############################################################

tstep = 0.001
NSteps = 100
Tmax = NSteps*tstep

############################################################
#Coordinate step and number of points for FFT
############################################################

limit = 30
mapping = 2*limit
nn = 2**12
delta = mapping/nn
f0 = 1/(nn*delta)

############################################################
#Initial wave package and coordinates
############################################################

x = []
momenta = []
Psi = [[],[]]


for i in range(nn):
    for j in range(2):
        x.append((i-(nn >> 1))*delta)
        if N < 0:
            Psi[0].append(_Psi((i-(nn >> 1))*delta,j))
            Psi[1].append(0)
        else:
            Psi[1].append(_Psi((i-(nn >> 1))*delta,j))
            Psi[0].append(0)

        if i < (nn >> 1):
            momenta.append(2*f0*i*math.pi)
        else:
            momenta.append(2*f0*(i-nn)*math.pi)

        
#I1 = _integration(nn,delta,[Psi[0][i]**2+Psi[0][i+1]**2 for i in range(nn << 1) if bin(i)[-1] == '0'])
#I2 = _integration(nn,delta,[Psi[1][i]**2+Psi[1][i+1]**2 for i in range(nn << 1) if bin(i)[-1] == '0'])

Integrals_1 = []
Integrals_2 = []

#############################################################
#Dynamic plotting
#############################################################

plt.ion()
fig = plt.figure(figsize = (10,15), facecolor = 'w')
ax = fig.add_axes([0.1,0.1,0.8,0.8])
line1, = ax.plot([x[i] for i in range(nn << 1) if bin(i)[-1] == '0'],#
                 [Psi[0][i]**2+Psi[0][i+1]**2 for i in range(nn << 1) if bin(i)[-1] == '0'])
line2, = ax.plot([x[i] for i in range(nn << 1) if bin(i)[-1] == '0'],#
                 [Psi[1][i]**2+Psi[1][i+1]**2 for i in range(nn << 1) if bin(i)[-1] == '0'])
ax.plot((x0,x0),(-0.05,1.),'k-',c = 'r')
#ax.set_xlim([-7,-2])
#ax.set_ylim([-0.,0.002])

#############################################################
#Propagation
#############################################################

V1 = []
V2 = []
rotations = []

for i in range(nn << 1):
    [v1,v2],aa = _diag(_potential(x[i],N))
    V1.append(v1)
    V2.append(v2)
    rotations.append(aa)

T = []
for i in range(nn << 1):
    T.append(_kinetic(momenta[i]))

    
t = 0

while t < Tmax:

############################################################
#1. Multiply on exponent with potential energy
############################################################

    for i in range(0,nn << 1,2):
        Psi[0][i]   = rotations[i][0]*Psi[0][i]-rotations[i][1]*Psi[1][i]
        Psi[1][i]   = rotations[i][1]*Psi[0][i]+rotations[i][0]*Psi[1][i]
        Psi[0][i+1] = rotations[i+1][0]*Psi[0][i+1]-rotations[i+1][1]*Psi[1][i+1]
        Psi[1][i+1] = rotations[i+1][1]*Psi[0][i+1]+rotations[i+1][0]*Psi[1][i+1]

        Psi[0][i]   = _Propagate(Psi[0][i],Psi[0][i+1],tstep,V1[i],i)
        Psi[0][i+1] = _Propagate(Psi[0][i],Psi[0][i+1],tstep,V1[i],i+1)

        Psi[1][i]   = _Propagate(Psi[1][i],Psi[1][i+1],tstep,V2[i],i)
        Psi[1][i+1] = _Propagate(Psi[1][i],Psi[1][i+1],tstep,V2[i],i+1)

        Psi[0][i]   = rotations[i][0]*Psi[0][i]+rotations[i][1]*Psi[1][i]
        Psi[1][i]   = -rotations[i][1]*Psi[0][i]+rotations[i][0]*Psi[1][i]
        Psi[0][i+1] = rotations[i+1][0]*Psi[0][i+1]+rotations[i+1][1]*Psi[1][i+1]
        Psi[1][i+1] = -rotations[i+1][1]*Psi[0][i+1]+rotations[i+1][0]*Psi[1][i+1]

############################################################
#2. Transform to impulse space
############################################################

    Psi[0] = _FFT(Psi[0],-1,nn,delta)
    Psi[0] = _FFT_norm(Psi[0],nn)

    Psi[1] = _FFT(Psi[1],-1,nn,delta)
    Psi[1] = _FFT_norm(Psi[1],nn)

#    line1.set_ydata([Psi[0][i]**2+Psi[0][i+1]**2 for i in range(nn << 1) if bin(i)[-1] == '0'])
#    line2.set_ydata([Psi[1][i]**2+Psi[1][i+1]**2 for i in range(nn << 1) if bin(i)[-1] == '0'])

############################################################
#3. Multiply on exponent with kinetic energy
############################################################

    for i in range(0,nn << 1,2):
        Psi[0][i]   = _Propagate(Psi[0][i],Psi[0][i+1],tstep,T[i],i)
        Psi[0][i+1] = _Propagate(Psi[0][i],Psi[0][i+1],tstep,T[i],i+1)

        Psi[1][i]   = _Propagate(Psi[1][i],Psi[1][i+1],tstep,T[i],i)
        Psi[1][i+1] = _Propagate(Psi[1][i],Psi[1][i+1],tstep,T[i],i+1)
        
############################################################
#4. Transform to coordinate space
############################################################
    
    Psi[0] = _FFT(Psi[0],1,nn,2*math.pi*f0)
    Psi[1] = _FFT(Psi[1],1,nn,2*math.pi*f0)

############################################################
#5. Normalization
############################################################

    for i in range(0,nn << 1,2):
        Psi[0][i]   = rotations[i][0]*Psi[0][i]-rotations[i][1]*Psi[1][i]
        Psi[1][i]   = rotations[i][1]*Psi[0][i]+rotations[i][0]*Psi[1][i]
        Psi[0][i+1] = rotations[i+1][0]*Psi[0][i+1]-rotations[i+1][1]*Psi[1][i+1]
        Psi[1][i+1] = rotations[i+1][1]*Psi[0][i+1]+rotations[i+1][0]*Psi[1][i+1]

    I1 = _integration(nn,delta,#
                     [Psi[0][i]**2+Psi[0][i+1]**2 for i in range(nn << 1) if bin(i)[-1] == '0'])
    I2 = _integration(nn,delta,#
                     [Psi[1][i]**2+Psi[1][i+1]**2 for i in range(nn << 1) if bin(i)[-1] == '0'])

    Integrals_1.append(I1)
    Integrals_2.append(I2)

    Norm = 1/math.sqrt(I1+I2)
    for i in range(0,nn << 1,2):
        Psi[0][i]   *= Norm
        Psi[0][i+1] *= Norm
        Psi[1][i]   *= Norm
        Psi[1][i+1] *= Norm

############################################################
#6. Nice plot
############################################################

    line1.set_ydata([Psi[0][i]**2+Psi[0][i+1]**2 for i in range(nn << 1) if bin(i)[-1] == '0'])
    line2.set_ydata([Psi[1][i]**2+Psi[1][i+1]**2 for i in range(nn << 1) if bin(i)[-1] == '0'])

    plt.pause(0.05)

    for i in range(0,nn << 1,2):
        Psi[0][i]   = rotations[i][0]*Psi[0][i]+rotations[i][1]*Psi[1][i]
        Psi[1][i]   = -rotations[i][1]*Psi[0][i]+rotations[i][0]*Psi[1][i]
        Psi[0][i+1] = rotations[i+1][0]*Psi[0][i+1]+rotations[i+1][1]*Psi[1][i+1]
        Psi[1][i+1] = -rotations[i+1][1]*Psi[0][i+1]+rotations[i+1][0]*Psi[1][i+1]

    
############################################################
#7. Next time step
############################################################

    t += tstep

#plt.plot([i*tstep for i in range(NSteps)],Integrals_1)        
#plt.plot([i*tstep for i in range(NSteps)],Integrals_2)        

#############################################################

print("--- %s seconds ---" % (time.time() - start_time))

############################################################
#<-- Main function ends here
############################################################
