# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 19:01:51 2017

@author: User
"""
import random
import math
import matplotlib.pyplot as plt
import time

A = [0.01,0.1,0.0006]
B = [1.6,0.28,0.1]
C = [0.005,0.015,0.9]
D = [1.,0.06]
E_0 = 0.05

mass = 2000
with open('./num_hops.txt','w') as fout:
    fout.write('')

#####################################################
#Potentials (with diag.) and there derivatives
#####################################################

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
    return V
   
def _potential_d(x,N):
    if N == 1:
        return math.sqrt(_potential(x,N)[0]**2+_potential(x,N)[2]**2)
    if N == 2:
        return (_potential(x,N)[1]+math.sqrt(_potential(x,N)[1]**2+4*_potential(x,N)[2]**2))/2
    if N == 3:
        return math.sqrt(_potential(x,N)[0]**2+_potential(x,N)[2]**2)
    if N == -1:
        return -math.sqrt(_potential(x,N)[0]**2+_potential(x,N)[2]**2)
    if N == -2:
        return (_potential(x,N)[1]-math.sqrt(_potential(x,N)[1]**2+4*_potential(x,N)[2]**2))/2
    if N == -3:
        return -math.sqrt(_potential(x,N)[0]**2+_potential(x,N)[2]**2)

def _diff_potential(x,N):
    dy = [0,0,0]    
    if N in [1,-1]:
        if x >= 0:
            dy[0] = A[0]*B[0]*math.exp(-B[0]*x)
        else:
            dy[0] = A[0]*B[0]*math.exp(B[0]*x)
        dy[1] = -dy[0]
        dy[2] = -2*C[0]*D[0]*x*math.exp(-D[0]*x**2)
    if N in [2,-2]:
        dy[0] = 0
        dy[1] = 2*A[1]*B[1]*x*math.exp(-B[1]*x**2)
        dy[2] = -2*C[1]*D[1]*x*math.exp(-D[1]*x**2)
    if N in [3,-3]:
        dy[0] = 0
        dy[1] = 0
        if x < 0:
            dy[2] = B[2]*C[2]*math.exp(C[2]*x)
        else:
            dy[2] = B[2]*C[2]*math.exp(-C[2]*x)
    return dy

def _diff_potential_d(x,N):
    if N == 1:
        return (_potential(x,N)[0]*_diff_potential(x,N)[0]+_potential(x,N)[2]*_diff_potential(x,N)[2])/math.sqrt(_potential(x,N)[0]**2+_potential(x,N)[2]**2)
    if N == -1:
        return -(_potential(x,N)[0]*_diff_potential(x,N)[0]+_potential(x,N)[2]*_diff_potential(x,N)[2])/math.sqrt(_potential(x,N)[0]**2+_potential(x,N)[2]**2)
    if N == 2:
        return _diff_potential(x,N)[1]*0.5+(_potential(x,N)[1]*_diff_potential(x,N)[1]+4*_potential(x,N)[2]*_diff_potential(x,N)[2])/(2*math.sqrt(_potential(x,N)[1]**2+4*_potential(x,N)[2]**2))
    if N == -2:
        return _diff_potential(x,N)[1]*0.5-(_potential(x,N)[1]*_diff_potential(x,N)[1]+4*_potential(x,N)[2]*_diff_potential(x,N)[2])/(2*math.sqrt(_potential(x,N)[1]**2+4*_potential(x,N)[2]**2))
    if N == 3:
        return (_potential(x,N)[0]*_diff_potential(x,N)[0]+_potential(x,N)[2]*_diff_potential(x,N)[2])/math.sqrt(_potential(x,N)[0]**2+_potential(x,N)[2]**2)
    if N == -3:
        return -(_potential(x,N)[0]*_diff_potential(x,N)[0]+_potential(x,N)[2]*_diff_potential(x,N)[2])/math.sqrt(_potential(x,N)[0]**2+_potential(x,N)[2]**2)

#####################################################
#Nonadiabatic coupling vector
#####################################################

def _d12(y,N):
    x = y[0]
    v1 = _potential(x,N)[0]
    v2 = _potential(x,N)[1]
    v12 = _potential(x,N)[2]
    dv1 = _diff_potential(x,N)[0]
    dv2 = _diff_potential(x,N)[1]
    dv12 = _diff_potential(x,N)[2]
    d12 = (dv12*(v2-v1)-v12*(dv2-dv1))/((v2-v1)**2+4*v12**2)
    return d12

#####################################################
#Random normal numbers
#####################################################

def _randnormal():
    U = random.uniform(0,1)
    V = random.uniform(0,1)
    z = math.sqrt(-2*math.log(U))*math.cos(2*math.pi*V) 
    return z

#####################################################    
#Equations of molecular dynamics
#####################################################

def _Verle_velocity(time,initial_cond,N,NTraj):
    R = [[0 for i in initial_cond[0]] for i in range(len(time))]
    P = [[0 for i in initial_cond[0]] for i in range(len(time))]
    E = [[0 for i in initial_cond[0]] for i in range(len(time))]
    for j in range(NTraj):
        R[0][j] = initial_cond[0][j]
        P[0][j] = initial_cond[1][j]
        E[0][j] = initial_cond[2][j]
    for i in range(len(time)-1):
        for j in range(NTraj):
            R[i+1][j] = R[i][j] + h*P[i][j]/mass - h**2/(2*mass)*_diff_potential_d(R[i][j],N)
            P[i+1][j] = P[i][j] - mass*h/2*(_diff_potential_d(R[i][j],N)+_diff_potential_d(R[i+1][j],N))
            E[i+1][j] = P[i+1][j]**2/(2*mass)+_potential_d(R[i+1][j],N)
            s = math.sqrt(2*mass*(E[0][j]-_potential_d(R[i+1][j],N))/P[i+1][j]**2)
            if abs(1-s) < 1*10**-4: pass
            else: 
                print(str(j)+' '+str(E[i+1][j])+' '+str(s)+' energy does not conserve')
                P[i+1][j] *= s
                E[i+1][j] = P[i+1][j]**2/(2*mass)+_potential_d(R[i+1][j],N)

    return R,P,E
    
def _Runge_Kutta(time,initial_cond,N,Ntraj):
    R = [[0 for i in initial_cond[0]] for i in range(len(time))]
    P = [[0 for i in initial_cond[0]] for i in range(len(time))]
    E = [[0 for i in initial_cond[0]] for i in range(len(time))]
    for j in range(NTraj):
        R[0][j] = initial_cond[0][j]
        P[0][j] = initial_cond[1][j]
        E[0][j] = initial_cond[2][j]
    for i in range(len(time)-1):
        for j in range(NTraj):
            R[i+1][j] = R[i][j] + h*P[i][j]/mass
            P[i+1][j] = P[i][j] - h*_diff_potential_d(R[i][j],N)

            R[i+1][j] = R[i][j] + h*(P[i][j]+P[i+1][j])/(2*mass)
            P[i+1][j] = P[i][j] - h*(_diff_potential_d(R[i][j],N)+_diff_potential_d(R[i+1][j],N))/2
            E[i+1][j] = P[i+1][j]**2/(2*mass)+_potential_d(R[i+1][j],N)
            s = math.sqrt(2*mass*(E[0][j]-_potential_d(R[i+1][j],N))/P[i+1][j]**2)
            if abs(1-s) < 1*10**-5: pass
            else: 
                print(str(j)+' '+str(E[i+1][j])+' '+str(s)+' energy does not conserve')
                P[i+1][j] *= s
                E[i+1][j] = P[i+1][j]**2/(2*mass)+_potential_d(R[i+1][j],N)
    return R,P,E

#####################################################
#Diffirential equations for populations
#####################################################
            
def _a_11_dot(y,N):
    x = y[0]
    rdot = y[1]
    u = y[4]
    v = y[5]
    return -2*v*_potential(x,N)[2]-2*u*rdot*_d12(y,N)

def _a_22_dot(y,N):
    x = y[0]
    rdot = y[1]
    u = y[4]
    v = y[5]
    return 2*v*_potential(x,N)[2]+2*u*rdot*_d12(y,N)

def _Re_a_12_dot(y,N):
    x = y[0]
    rdot = y[1]
    a_11 = y[2]
    a_22 = y[3]
    v = y[5]
    return v*(_potential(x,N)[0]-_potential(x,N)[1])-rdot*_d12(y,N)*(a_22-a_11)

def _Im_a_12_dot(y,N):
    x = y[0]
    a_11 = y[2]
    a_22 = y[3]
    u = y[4]
    return -u*(_potential(x,N)[0]-_potential(x,N)[1])-_potential(x,N)[2]*(a_22-a_11)

#####################################################
#Function to solve dynamics with hops
#####################################################
##Second order implicit Runge-Kutta
#####################################################

def _hop_probability(time,h,initial_cond,N,Ntraj):
    R = [[0 for i in initial_cond[0]] for i in range(len(time))]
    P = [[0 for i in initial_cond[0]] for i in range(len(time))]
    E = [[0 for i in initial_cond[0]] for i in range(len(time))]
    a_11 = [[0 for i in initial_cond[0]] for i in range(len(time))]
    a_22 = [[0 for i in initial_cond[0]] for i in range(len(time))]
    Re_a_12 = [[0 for i in initial_cond[0]] for i in range(len(time))]
    Im_a_12 = [[0 for i in initial_cond[0]] for i in range(len(time))]
    v = [[0 for i in initial_cond[0]] for i in range(len(time))]
    
    for j in range(NTraj):
        R[0][j] = initial_cond[0][j]
        P[0][j] = initial_cond[1][j]
        E[0][j] = initial_cond[2][j]
        a_11[0][j] = initial_cond[3][j]
        a_22[0][j] = initial_cond[4][j]
        Re_a_12[0][j] = initial_cond[5][j]
        Im_a_12[0][j] = initial_cond[6][j]
        v[0][j] = _potential_d(R[0][j],N[j])
    for j in range(NTraj):
        dzeta = random.uniform(0,1)
        for i in range(len(time)-1):        
            velocity1 = P[i][j]/mass
            vec1 = [R[i][j],velocity1,a_11[i][j],a_22[i][j],Re_a_12[i][j],Im_a_12[i][j]]
            v[i+1][j] = _potential_d(R[i][j],N[j])

            R[i+1][j] = R[i][j] + h*velocity1
            P[i+1][j] = P[i][j] - h*_diff_potential_d(R[i][j],N[j])
            a_11[i+1][j] = a_11[i][j]+h*_a_11_dot(vec1,N[j])
            a_22[i+1][j] = a_22[i][j]+h*_a_22_dot(vec1,N[j])
            Re_a_12[i+1][j] = Re_a_12[i][j]+h*_Re_a_12_dot(vec1,N[j])
            Im_a_12[i+1][j] = Im_a_12[i][j]+h*_Im_a_12_dot(vec1,N[j])

            velocity2 = P[i+1][j]/mass
            vec2 = [R[i+1][j],P[i+1][j]/mass,a_11[i+1][j],a_22[i+1][j],Re_a_12[i+1][j],Im_a_12[i+1][j]]

            R[i+1][j] = R[i][j] + h*(velocity1+velocity2)*0.5
            P[i+1][j] = P[i][j] - h*(_diff_potential_d(R[i][j],N[j])+_diff_potential_d(R[i+1][j],N[j]))*0.5
               
            a_11[i+1][j] = a_11[i][j]+h*(_a_11_dot(vec1,N[j])+_a_11_dot(vec2,N[j]))*0.5
            a_22[i+1][j] = a_22[i][j]+h*(_a_22_dot(vec1,N[j])+_a_22_dot(vec2,N[j]))*0.5
            Re_a_12[i+1][j] = Re_a_12[i][j]+h*(_Re_a_12_dot(vec1,N[j])+_Re_a_12_dot(vec2,N[j]))*0.5
            Im_a_12[i+1][j] = Im_a_12[i][j]+h*(_Im_a_12_dot(vec1,N[j])+_Im_a_12_dot(vec2,N[j]))*0.5

            if N[j] < 0:
                g = h*_a_22_dot(vec2,N[j])/a_11[i+1][j]
            else:
                g = h*_a_11_dot(vec2,N[j])/a_22[i+1][j]
            if dzeta > g:
                pass
            if dzeta < g:
                s = P[i][j]**2+2*mass*(_potential_d(R[i][j],N[j])-_potential_d(R[i+1][j],-N[j]))
                if s >= 0:
                    N[j] /= -1
                    with open('./num_hops.txt','a') as fout:
                        fout.write('trajectrory: '+str(j)+' hop from: ' +str(-N[j])+ ' to: '+str(N[j])+' at: '+str(round(R[i][j],2))+' '+str(round(time[i],2))+'\n')
                else: 
                    pass

            E[i+1][j] = P[i+1][j]**2*0.5/mass+_potential_d(R[i+1][j],N[j])
            if abs(1-E[i+1][j]/E[0][j]) < 1*10**-2: pass
            else: 
                print(str(j)+' '+str(round(E[i+1][j],4))+' '+str(round(E[i+1][j]/E[0][j],4)) + ' energy does not conserve')
                P[i+1][j] = math.sqrt(s)
                E[i+1][j] = P[i+1][j]**2/(2*mass)+_potential_d(R[i+1][j],N[j])

    return R,P,E,a_11,a_22,Re_a_12,Im_a_12,v

#####################################################
##Fourth order implicit Runge-Kutta
#####################################################

def _hop_probability_1(time,h,initial_cond,N,Ntraj):
    R = [[0 for i in initial_cond[0]] for i in range(len(time))]
    P = [[0 for i in initial_cond[0]] for i in range(len(time))]
    E = [[0 for i in initial_cond[0]] for i in range(len(time))]
    a_11 = [[0 for i in initial_cond[0]] for i in range(len(time))]
    a_22 = [[0 for i in initial_cond[0]] for i in range(len(time))]
    Re_a_12 = [[0 for i in initial_cond[0]] for i in range(len(time))]
    Im_a_12 = [[0 for i in initial_cond[0]] for i in range(len(time))]
    v = [[0 for i in initial_cond[0]] for i in range(len(time))]

    for j in range(NTraj):
        R[0][j] = initial_cond[0][j]
        P[0][j] = initial_cond[1][j]
        E[0][j] = initial_cond[2][j]
        a_11[0][j] = initial_cond[3][j]
        a_22[0][j] = initial_cond[4][j]
        Re_a_12[0][j] = initial_cond[5][j]
        Im_a_12[0][j] = initial_cond[6][j]
        v[0][j] = _potential_d(R[0][j],N[j])
        for i in range(2):
            vec1 = [R[i][j],P[i][j]/mass,a_11[i][j],a_22[i][j],Re_a_12[i][j],Im_a_12[i][j]]

            R[i+1][j] = R[i][j] + h*P[i][j]/mass
            P[i+1][j] = P[i][j] - h*_diff_potential_d(R[i][j],N[j])
            a_11[i+1][j] = a_11[i][j]+h*_a_11_dot(vec1,N[j])
            a_22[i+1][j] = a_22[i][j]+h*_a_22_dot(vec1,N[j])
            Re_a_12[i+1][j] = Re_a_12[i][j]+h*_Re_a_12_dot(vec1,N[j])
            Im_a_12[i+1][j] = Im_a_12[i][j]+h*_Im_a_12_dot(vec1,N[j])

            vec2 = [R[i+1][j],P[i+1][j]/mass,a_11[i+1][j],a_22[i+1][j],Re_a_12[i+1][j],Im_a_12[i+1][j]]

            R[i+1][j] = R[i][j] + h*(P[i][j]+P[i+1][j])*0.5/mass
            P[i+1][j] = P[i][j] - h*(_diff_potential_d(R[i][j],N[j])+_diff_potential_d(R[i+1][j],N[j]))*0.5

            a_11[i+1][j] = a_11[i][j]+h*(_a_11_dot(vec1,N[j])+_a_11_dot(vec2,N[j]))*0.5
            a_22[i+1][j] = a_22[i][j]+h*(_a_22_dot(vec1,N[j])+_a_22_dot(vec2,N[j]))*0.5
            Re_a_12[i+1][j] = Re_a_12[i][j]+h*(_Re_a_12_dot(vec1,N[j])+_Re_a_12_dot(vec2,N[j]))*0.5
            Im_a_12[i+1][j] = Im_a_12[i][j]+h*(_Im_a_12_dot(vec1,N[j])+_Im_a_12_dot(vec2,N[j]))*0.5

            E[i+1][j] = P[i+1][j]**2*0.5/mass+_potential_d(R[i+1][j],N[j])
            v[i+1][j] = _potential_d(R[i+1][j],N[j])

    for j in range(NTraj):
        dzeta = random.uniform(0,1)
        for i in range(2,len(time)-1):        
                R[i+1][j] = R[i][j] + h*(23*P[i][j]-16*P[i-1][j]+5*P[i-2][j])/(12*mass)
                P[i+1][j] = P[i][j] - h*(23*_diff_potential_d(R[i][j],N[j])-#
                                        16*_diff_potential_d(R[i-1][j],N[j])+#
                                        5*_diff_potential_d(R[i-2][j],N[j]))/12
                
                vec1 = [R[i][j],P[i][j]/mass,a_11[i][j],#
                        a_22[i][j],Re_a_12[i][j],Im_a_12[i][j]]
                vec2 = [R[i-1][j],P[i-1][j]/mass,a_11[i-1][j],#
                        a_22[i-1][j],Re_a_12[i-1][j],Im_a_12[i-1][j]]
                vec3 = [R[i-2][j],P[i-2][j]/mass,a_11[i-2][j],#
                        a_22[i-2][j],Re_a_12[i-2][j],Im_a_12[i-2][j]]

                a_11[i+1][j] = a_11[i][j] + h*(23*_a_11_dot(vec1,N[j])-#
                                               16*_a_11_dot(vec2,N[j])+#
                                               5*_a_11_dot(vec3,N[j]))/12
                a_22[i+1][j] = a_22[i][j] + h*(23*_a_22_dot(vec1,N[j])-#
                                               16*_a_22_dot(vec2,N[j])+#
                                               5*_a_22_dot(vec3,N[j]))/12
                Re_a_12[i+1][j] = Re_a_12[i][j] + h*(23*_Re_a_12_dot(vec1,N[j])-#
                                               16*_Re_a_12_dot(vec2,N[j])+#
                                               5*_Re_a_12_dot(vec3,N[j]))/12
                Im_a_12[i+1][j] = Im_a_12[i][j] + h*(23*_Im_a_12_dot(vec1,N[j])-#
                                               16*_Im_a_12_dot(vec2,N[j])+#
                                               5*_Im_a_12_dot(vec3,N[j]))/12

                vec4 = [R[i+1][j],P[i+1][j]/mass,a_11[i+1][j],#
                        a_22[i+1][j],Re_a_12[i+1][j],Im_a_12[i+1][j]]
                
                R[i+1][j] = R[i][j] + h*(9*P[i+1][j]+19*P[i][j]-#
                                         5*P[i-1][j]+P[i-2][j])/(12*mass)
                P[i+1][j] = P[i][j] - h*(9*_diff_potential_d(R[i+1][j],N[j])+#
                                         19*_diff_potential_d(R[i][j],N[j])-#
                                         5*_diff_potential_d(R[i-1][j],N[j])+#
                                         _diff_potential_d(R[i-2][j],N[j]))/12

                a_11[i+1][j] = a_11[i][j] + h*(9*_a_11_dot(vec4,N[j])+#
                                               19*_a_11_dot(vec1,N[j])-#
                                               5*_a_11_dot(vec2,N[j])+#
                                               _a_11_dot(vec3,N[j]))/12
                a_22[i+1][j] = a_22[i][j] + h*(9*_a_22_dot(vec4,N[j])+#
                                               19*_a_22_dot(vec1,N[j])-#
                                               5*_a_22_dot(vec2,N[j])+#
                                               _a_22_dot(vec3,N[j]))/12
                Re_a_12[i+1][j] = Re_a_12[i][j] + h*(9*_Re_a_12_dot(vec4,N[j])+#
                                                     19*_Re_a_12_dot(vec1,N[j])-#
                                                     5*_Re_a_12_dot(vec2,N[j])+#
                                                     _Re_a_12_dot(vec3,N[j]))/12
                Im_a_12[i+1][j] = Im_a_12[i][j] + h*(9*_Im_a_12_dot(vec4,N[j])+#
                                                     19*_Im_a_12_dot(vec1,N[j])-#
                                                     5*_Im_a_12_dot(vec2,N[j])+#
                                                     _Im_a_12_dot(vec3,N[j]))/12

                vec4 = [R[i+1][j],P[i+1][j]/mass,a_11[i+1][j],#
                        a_22[i+1][j],Re_a_12[i+1][j],Im_a_12[i+1][j]]

                if N[j] < 0:
                    g = h*_a_22_dot(vec4,N[j])/a_11[i+1][j]
                else:
                    g = h*_a_11_dot(vec4,N[j])/a_22[i+1][j]
                if dzeta > g:
                    pass
                if dzeta < g:
                    s = P[i-1][j]**2+2*mass*(_potential_d(R[i-1][j],N[j])-_potential_d(R[i+1][j],-N[j]))
                    if s >= 0:
                        N[j] /= -1
                        with open('./num_hops.txt','a') as fout:
                            fout.write('trajectrory: '+str(j)+' hop from: ' +str(-N[j])+ ' to: '+str(N[j])+' at: '+str(round(R[i][j],2))+' '+str(round(time[i],2))+'\n')
                    else: 
                        pass

                v[i+1][j] = _potential_d(R[i+1][j],N[j])
                E[i+1][j] = P[i+1][j]**2*0.5/mass+_potential_d(R[i+1][j],N[j])
                if abs(1-E[i+1][j]/E[i-1][j]) < 1*10**-2: pass
                else: 
                    print(str(j)+' '+str(round(E[i+1][j],4))+' '+str(round(E[i+1][j]/E[0][j],4)) + ' energy does not conserve')
                    P[i+1][j] = math.sqrt(P[i-1][j]**2+2*mass*(_potential_d(R[i-1][j],-N[j])-_potential_d(R[i+1][j],N[j])))
                    E[i+1][j] = P[i+1][j]**2/(2*mass)+_potential_d(R[i+1][j],N[j])

    return R,P,E,a_11,a_22,Re_a_12,Im_a_12,v

#####################################################

#####################################################
#Time estimation
#####################################################
            
start_time = time.time()

#####################################################
#Number of trajectories
#####################################################

NTraj = 10

#####################################################
#Potential
#####################################################

N = []
for i in range(NTraj):
    N.append(-2)

#####################################################
#Initial conditions
#####################################################

r_mean = -10
p_mean = 9
E_mean = p_mean**2/(2*mass)+_potential_d(r_mean,N[0])

sigma_r = 0.5
sigma_p = 2

R0 = []
P0 = []
a11_0 = []
a22_0 = []
Re_a12_0 = []
Im_a12_0 = []
for i in range(NTraj):
    R0.append(r_mean+sigma_r*_randnormal())
    P0.append(p_mean+sigma_p*_randnormal())
    Re_a12_0.append(0)
    Im_a12_0.append(0)
    if N[i] < 0:
        a11_0.append(1)
        a22_0.append(0)
    else:
        a11_0.append(0)
        a22_0.append(1)

E0 = [p**2/(2*mass)+_potential_d(r,n) for r,p,n in zip(R0,P0,N)]

#####################################################
#Plot potential, analitical and numerical derivatives
#####################################################

#for j in range(NTraj):
#    X = [i/10 for i in range(-100,100)]
#    plt.plot(X,[_potential_d(r,N[j]) for r in X])
#    plt.plot(X,[_diff_potential_d(r,N[j]) for r in X])
#    plt.plot([X[i] for i in range(len(X)-1)],[(_potential_d(X[i+1],N[j])-_potential_d(X[i],N[j]))/(X[i+1]-X[i]) for i in range(len(X)-1)])

#####################################################
#Array of Time points
#####################################################

Tmax = 5000
h = 2
Time = []
t = 0
while t <= Tmax:
    t += h
    Time.append(t)

#####################################################
#Solve dynamical equations (Verley or Runge-Kutta)
#####################################################

#R,P,E = _Verle_velocity(Time,[R0,P0,E0],N,NTraj)
#R,P,E = _Runge_Kutta(Time,[R0,P0,E0],N,NTraj)

#####################################################
#Nice plots
#####################################################

#for j in range(NTraj):
#    plt.plot([R[i][j] for i in range(int(Tmax//h))],[P[i][j] for i in range(int(Tmax//h))])

#####################################################
#Mean values
#####################################################

#R_mean = [0 for i in range(len(Time))]
#P_mean = [0 for i in range(len(Time))]
#E_mean = [0 for i in range(len(Time))]
#
#for i in range(len(Time)):
#    for j in range(NTraj):
#        R_mean[i] += R[i][j]
#        P_mean[i] += P[i][j]
#        E_mean[i] += E[i][j]
#        
#    R_mean[i] /= NTraj
#    P_mean[i] /= NTraj
#    E_mean[i] /= NTraj
#    
#plt.plot(Time,P_mean)

#####################################################
#Solve extended system of ODE with density
#####################################################

init = [R0,P0,E0,a11_0,a22_0,Re_a12_0,Im_a12_0]
R,P,E,a11,a22,Rea12,Ima12,v_eff = _hop_probability_1(Time,h,init,N,NTraj)


fig1 = plt.figure(figsize = (10,10),facecolor = 'w')
ax1 = fig1.add_axes([0.1,0.1,0.8,0.8])

for j in range(NTraj):
    ax1.plot(Time,[R[i][j] for i in range(len(Time))])

fig2 = plt.figure(figsize = (10,10),facecolor = 'w')
ax2 = fig2.add_axes([0.1,0.1,0.8,0.8])

for j in range(NTraj):
#    ax2.plot(Time,[_d12([R[i][j]],N) for i in range(len(Time))])
    ax2.plot([R[i][j] for i in range(len(Time))],[v_eff[i][j] for i in range(len(Time))])

#####################################################
#Populations
#####################################################

#a_1_d = [[] for i in range(NTraj)]
#a_2_d = [[] for i in range(NTraj)]

#for j in range(NTraj):
#    for i in range(len(Time)):
#        a_1_d[j].append((a11[i][j]+a22[i][j]+math.sqrt((a11[i][j]-a22[i][j])**2+4*(Rea12[i][j]**2+Ima12[i][j]**2)))/2)
#        a_2_d[j].append((a11[i][j]+a22[i][j]-math.sqrt((a11[i][j]-a22[i][j])**2+4*(Rea12[i][j]**2+Ima12[i][j]**2)))/2)

#for j in range(NTraj):
#    plt.plot(Time,a_1_d[j])
#    plt.plot(Time,a_2_d[j])

#####################################################
#Time estimation
#####################################################

print("--- %s seconds ---" % (time.time() - start_time))

#####################################################