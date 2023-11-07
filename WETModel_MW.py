# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 19:52:45 2022

@author: jdc478
"""

import numpy as np

e = 0.95
sigma = 5.67e-8
Ti = 100

ei = 0.05
er = 3.145
erw = 3.12 #Matsuoka 2014
eiw = 10e-4
e0 = 8.85e-12
cv = 3e8
f = 2.45e9


lamda0 = cv/f
lamdae = lamda0/(er)**0.5

Z0 = 377/(er)**0.5

#heat capacity coefficients
c0 = -2.32e-2
c1 = 2.13e-3
c2 = 1.5e-5
c3 = -7.37e-8
c4 = 9.66e-11

#time and spatial dimensions
dx = 0.01
dy= 0.01
dz = 0.01
dt = 1
Lx = 0.25
Ly = 0.25
Lz = 0.25

#system arrays
x_array = np.arange(0,Lx+dx, dx)
y_array = np.arange(0,Ly+dy,dy)
z_array = np.arange(0,Lz+dz, dz)

xend = len(x_array)-1  
yend = len(y_array)-1
zend = len(z_array)-1

E0 = np.ones((len(y_array),len(x_array),len(z_array)))
E1 = np.ones((len(z_array)))

eir = np.ones((len(y_array),len(x_array),len(z_array)))
err = np.ones((len(y_array),len(x_array),len(z_array)))

T = np.ones((len(y_array),len(x_array),len(z_array)))*Ti
Tnew = T.copy()

p1 =0.5
np.random.seed(1)
w = np.random.choice([0, 1, 1], size=(len(y_array),len(x_array),len(z_array)), p=[p1, (1-p1)/2, (1-p1)/2])

w[:,:,zend-2:] = 0
w[:,:3,:] = 0
w[:,xend-2:,:] = 0
w[:3,:,:] = 0
w[yend-2:,:,:] = 0   
w[:,:,:3] = 0   

wi = w.copy()
wcountlist = []
wcountinitial = sum(sum(sum(wi)))/(len(x_array)*len(y_array)*len(z_array))

c = np.ones([len(y_array),len(x_array),len(z_array)])*500
k = np.ones([len(y_array),len(x_array),len(z_array)])*0.3
dk = np.ones([len(y_array),len(x_array),len(z_array)])*0.3
rho = np.ones([len(y_array),len(x_array),len(z_array)])*1500
alpha = np.ones([len(y_array),len(x_array),len(z_array)])*1e-7

s= np.ones([len(y_array),len(x_array),len(z_array)])
#mloss = np.zeros([len(y_array),len(x_array),len(z_array)])
mltrack = np.zeros([len(y_array),len(x_array),len(z_array)])
m = np.ones([len(y_array),len(x_array),len(z_array)])*920*dx*dy*dz*1000
P = np.ones([len(y_array),len(x_array),len(z_array)])
mres = 1e-3

err = np.ones((len(y_array),len(x_array),len(z_array)))
eir = np.ones((len(y_array),len(x_array),len(z_array)))
x0 = 0.06
pi = np.pi

watertrack = []

def reset(p1, SEED):
    
    global w, E0, E1, wcountinitial, c, k, dk, rho, alpha, s, mltrack, m, P, Tnew, wcountlist, wi, wmask

    E0 = np.ones((len(y_array),len(x_array),len(z_array)))
    E1 = np.ones((len(z_array)))
    
    err = np.ones((len(y_array),len(x_array),len(z_array)))
    eir = np.ones((len(y_array),len(x_array),len(z_array)))
    
    T = np.ones((len(y_array),len(x_array),len(z_array)))*Ti
    Tnew = T.copy()
  
    np.random.seed(SEED)
    w = np.random.choice([0, 1, 1], size=(len(y_array),len(x_array),len(z_array)), p=[p1, (1-p1)/2, (1-p1)/2])
    
    w[:,:,zend-2:] = 0
    w[:,:3,:] = 0
    w[:,xend-2:,:] = 0
    w[:3,:,:] = 0
    w[yend-2:,:,:] = 0   
    w[:,:,:3] = 0   
        
    wi = w.copy()
    wcountlist = []
    wcountinitial = sum(sum(sum(wi)))/(len(x_array)*len(y_array)*len(z_array))

    c = np.ones([len(y_array),len(x_array),len(z_array)])*500
    k = np.ones([len(y_array),len(x_array),len(z_array)])*0.3
    dk = np.ones([len(y_array),len(x_array),len(z_array)])*0.3
    rho = np.ones([len(y_array),len(x_array),len(z_array)])*1500
    alpha = np.ones([len(y_array),len(x_array),len(z_array)])*1e-7
    
    s= np.ones([len(y_array),len(x_array),len(z_array)])
    #mloss = np.zeros([len(y_array),len(x_array),len(z_array)])
    mltrack = np.zeros([len(y_array),len(x_array),len(z_array)])
    m = np.ones([len(y_array),len(x_array),len(z_array)])*920*dx*dy*dz*1000
    P = np.ones([len(y_array),len(x_array),len(z_array)])
    mres = 1e-3
    
    watertrack.append(str(round(wcountinitial*100,1)))

    return w, E0, E1, w, wcountinitial, c, k, dk, rho, alpha, s, mltrack, m, P, Tnew, wcountlist, p1, wi, err, eir

def dielectricconstant(y,x,z,T):
    Td = Tnew[y,x,z]
    if Td <= 350:
        a = 3.75
        b = 1.923e-3
        ecalc = a + b*(Td-83)
        err[y,x,z] = ecalc
    else: 
        a = 5.52895e-10
        b = 2.80378e-6
        c = -2.83249e-3
        d = 4.89315
        ecalc = a*Td**3+b*Td**2+c*Td+d
        err[y,x,z] = ecalc
    return err

def dielectricloss(y,x,z,T):
    Td = Tnew[y,x,z]
    if Td <= 296:
        a = 8.42684e-11
        b = -9.34746e-8
        c = 3.39475e-5
        d = -4.46073e-3
        e = 0.202337
        ecalc = a*(Td**4)+b*(Td**3)+c*(Td**2)+d*(Td)+e
        eir.append(ecalc)
    else: 
        a = -8.05191e-13
        b = 5.75914e-9
        c = -7.40846e-6
        d = 3.43871e-3
        e = -0.42633
        ecalc = a*Td**4+b*Td**3+c*Td**2+d*Td+e
        eir[y,x,z] = ecalc
    return err
        
def E_abs(ef,y,x, z,t):
    E1[z] = ef*np.cos(((2*pi*z*dz/lamdae))-(2*pi*2.45e9*t*dt))

def powerabs(y,x,z):
    if y > 4 and y < 21:
        if x > 4 and x < 21:
            if w[y,x,z] == 1:
                   dPaltw = (0.125*(erw)**0.5)/(2*3.14*eiw)  
                   P0 = 2*3.14*f*eiw*e0*(E1[z])**2
                   P[y,x, z] = P0*np.exp(-2*z*dz/dPaltw)
                    
            else:
                    dPaltr = (0.125*(err[y,x,z])**0.5)/(2*3.14*eir[y,x,z])  
                    P0 = 2*3.14*f*eir[y,x,z]*e0*(E1[z])**2
                    P[y,x, z] = P0*np.exp(-2*z*dz/dPaltr)          
    else:
        P[y,x,z] = 0

    return P

def heatcap(y,x,z, T):
    Td = T[y,x,z]
    if w[y,x,z] == 1:
        if Td <= 95:
            c[y,x,z] = -0.8994+(0.1710*Td)
            c[y,x,z] = c[y,x,z]/(18e-3) 
            
        elif Td > 95 and Td <= 150:
            c[y,x,z] =2.2841+(0.1350*Td)
            c[y,x,z] = c[y,x,z]/(18e-3) 

        else:
            c[y,x,z]= 2.7442+(0.1282*Td)
            c[y,x,z] = c[y,x,z]/(18e-3)
    
    else:
        if T[y,x,z] <= 350:
            c[y,x,z] = (c0+c1*T[y,x,z]+c2*(T[y,x,z])**2+c3*(T[y,x,z])**3+c4*(T[y,x,z])**4)*1000
        else:
            c[y,x,z] = ((9.530e2)+(2.524e-1*T[y,x,z])-(2.645e7*(T[y,x,z])**-2))
        
            
 
       
        
    return c

def thermcond(y,x,z,T):

    Td = T[y,x,z]

    if w[y,x,z] == 1:
        k[y,x,z] = (632/Td)+0.38+(-0.00197*Td)
     
    else:                   
        if Td < 200:
            k[y,x,z] = (4.0486e-10*Td**3)+(4.6278e-61*Td**2)+(1.5295e-6*Td)+1.2810e-2

        elif Td >= 200 and Td < 400:
            k[y,x,z] = (6.3429e-10*Td**3)+(-1.3766e-7*Td**2)+(2.9061e-5*Td)+1.0975e-2
        elif Td >= 400 and Td < 600:
            k[y,x,z] = (-2.8341e-10*Td**3)+(9.6359e-7*Td**2)+(-4.1144e-4*Td)+6.9707e-2
        elif Td >= 600 and Td < 800:
            k[y,x,z] = (6.5796e-10*Td**3)+(-7.3089e-7*Td**2)+(6.0525e-4*Td)-1.3363e-1
        elif Td >= 800 and Td < 1000:
            k[y,x,z] = (2.1852e-9*Td**3)+(-4.3961e-6*Td**2)+(3.5375e-3*Td)-9.1555e-1
        else:
            k[y,x,z] = (3.8850e-9*Td**3)+(-9.4957e-6*Td**2)+(8.6371e-3*Td)-2.6154


       
       
          
    return k

def density_initial(y,x,z):
  
    if w[y,x,z] ==1:
        rho[y,x,z] = 920

    else:
        rho[y,x,z] = 1700
   
    return rho,

def mass_initial(y,x,z):
    

    rtemp = rho[y,x,z]
    mtemp = (rtemp)*dx*dy*dz*1000
    m[y,x,z] = mtemp

  
    return m,

def density_dynamic(y,x,z):
  
    if w[y,x,z] == 2:
        rho_sum = (rho[y+1,x,z]+rho[y-1,x,z]+rho[y,x+1,z]+rho[y,x-1,z]+rho[y,x,z-1]+rho[y,x,z+1])/7
        rho[y,x,z] = rho_sum
        rho[y+1,x,z]=rho_sum
        rho[y-1,x,z]=rho_sum
        rho[y,x+1,z]=rho_sum
        rho[y,x-1,z]=rho_sum
        rho[y,x,z-1]=rho_sum
        rho[y,x,z+1]= rho_sum
        w[y,x,z]=3
        
        
    else:
        pass
    return rho

def alphacalc(y,x,z, t):
     
    alpha[y,x,z] = k[y,x,z]/(rho[y,x,z]*(c[y,x,z])) #heat diffusion factor

    return alpha

                

def sublimation(y,x,z, T):
    if w[y,x,z]==1:
        
        saturation(y,x,z, T)
        mass(y,x,z,s, T)
                        
                

    
def saturation(y,x,z, T):
    if T[y,x,z] > 273.15:
        m[y,x,z] = 0
        w[y,x,z] =2
        mltrack[y,x,z] = 920*dx*dy*dz*1000  
    else:
        
        s[y,x,z] = np.exp(9.550426-(5723.265/T[y,x,z])+3.53068*np.log(T[y,x,z])-0.00728332*T[y,x,z])

    return s
def mass(y,x,z, s, T):
    
    flux = s[y,x,z]*(18e-3/(2*3.14*8.31*T[y,x,z]))**0.5
    
    V = m[y,x,z]/(1000*rho[y,x,z]) #m3
    r = (V/((4/3)*3.14))**1/3 #m
    sa = 4*3.14*r**2 # m2

    mloss = flux*dt*1000*sa #g
    

    m[y,x,z] = m[y,x,z] - mloss
    mltrack[y,x,z] = mltrack[y,x,z] + mloss
    return m, mloss, mltrack



def BoundCondition(y,x,z,T):
    if z == 0 or z == zend:
        if z==0:
            
            Tnew[y,x,0] = T[y,x,0] + (dt/(c[y,x,0]*rho[y,x,0]))*(1/(dx*dy*dz))*((P[y,x,0]*dx*dy*dz)-sigma*e*((T[y,x,0])**4-40**4)*dx*dy)
            if Tnew[y,x,z] > 1000:
                Tnew[y,x,z] = 1000

        elif z == zend:
        
            Tnew[y,x,zend] = T[y,x,zend] 
            if Tnew[y,x,z] > 1000:
                Tnew[y,x,z] = 1000
        
    else:
        if y ==0:
            
            Tnew[0,x,z] = T[0,x,z] 
            if Tnew[y,x,z] > 1000:
                Tnew[y,x,z] = 1000
        elif x ==0:
            
            Tnew[y,0,z] = T[y,0,z] 
            if Tnew[y,x,z] > 1000:
                Tnew[y,x,z] = 1000
       
        elif y ==yend:
            
            Tnew[yend,x,z] = T[yend,x,z] 
            if Tnew[y,x,z] > 1000:
                Tnew[y,x,z] = 1000
                
        elif x == xend:
            
            Tnew[y,xend,z] = T[y,xend,z] 
            if Tnew[y,x,z] > 1000:
                Tnew[y,x,z] = 1000
            
        
        
    return T
   
def changetemp(y,x,z,f1, T, Tnew):
    

    
    deltaT = f1[y,x,z]*(T[y+1,x,z]+T[y,x+1,z]+T[y,x,z+1]+T[y-1,x,z]+T[y,x-1,z]+T[y,x,z-1]-6*T[y,x,z])+T[y,x,z]

    Tnew[y,x,z] = deltaT +(P[y,x,z]*dt)/(rho[y,x,z]*c[y,x,z])
    
    if Tnew[y,x,z] > 1000:
       Tnew[y,x,z] = 1000

    
    return Tnew


efftrack1 = []
Ttrack = []
timetrack = {}



def efficiency(t, ef, p1):
    InitialWater  = sum(sum(sum(wi)))*920*dx*dy*dz*1000
    print(InitialWater)
    MassLost = sum(sum(sum(mltrack)))

    Eff = MassLost*100/InitialWater
    
 
    if p1 == 0.5:
        efftrack1.append(Eff)
        timetrack[f'Time_{t}']=t
        print(Eff)
    else:
        efftrack1.append(Eff)
        Ttracker = Tnew.copy()
        Ttrack.append(Ttracker)
        print(Eff)

 




def calculate(ef, time, T, p1, SEED):

    t_array = np.arange(0,time+dt, dt)
    for y in range(0, len(y_array)):
        for x in range(0,len(x_array)):
            for z in range(0,len(z_array)):
                density_initial(y,x,z) #sets up initial density field
                mass_initial(y,x,z) #sets up inital mass field
               
        
    for t in t_array:
      
 
          
        for y in range(0,len(y_array)):
            for x in range(0,len(x_array)):
                for z in range(0,len(z_array)):
                        BoundCondition(y,x,z,T)
                        dielectricconstant(y,x,z, T)
                        dielectricloss(y,x,z,T)
                        E_abs(ef,y,x,z,t)
                        powerabs(y,x,z)
                        density_dynamic(y,x,z)
                        thermcond(y,x,z, T)
                        heatcap(y,x,z, T)
                        
                        sublimation(y,x,z,T)
                        
                        alphacalc(y,x,z, t)
                    
        if t%100 == 0:
            print('SEED: ' +str(SEED)+', Electric Field: ' +str(ef)+'V/m, Water Conc.: ' +str(round(wcountinitial*100,1))+'%, Total Time: ' +str(time)+'s, Running.... '+str(t)+'s')
    
            efficiency(t, ef, p1)
       
        
      
        f1 = (alpha[:,:,:]*dt)/dx**2
     
        for y in range(1, yend):
            for x in range(1, xend):
                for z in range(1, zend):
   
                    changetemp(y,x,z,f1, T, Tnew)     
    
        
        T = Tnew.copy()


ef = 8680 #Electric Field in V/m, calculated from repsective input power
TIME = 100 #time of running in seconds
SEED = 1 #water distribution seed
p1=0.5 #water content factor
  
reset(p1, SEED) #resets the system
calculate(ef, TIME, T,p1, SEED) #simulates the particular heating run

