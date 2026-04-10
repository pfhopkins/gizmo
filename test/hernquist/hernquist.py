#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import brentq
import h5py
from numba import jit
from sys import argv

G = 1 # gravitational constant


 # arguments: (0) number of particles (default 32^3) (1) total mass (default 1.0) (2) scale radius (default 1.0)
if len(argv)>1:
    N = int(float(argv[1])+0.5)
else:
    N = 32**3
if len(argv)>2:
    m = float(argv[2])
else:
    m = 1.
if len(argv)>3:
    a = float(argv[3])
else:
    a = 1.

x= np.arange(0,1,1./N) + np.random.rand(N)/N

r = (np.sqrt(x)+x)/(1 - x) #np.sqrt(x**(2./3) * (1+x**(2./3) + x**(4./3))/(1-x**2))
phi = np.random.rand(N) * 2 * np.pi
theta = np.arccos(2.0*np.random.rand(N)-1.0)
plt.loglog(np.sort(r), (np.arange(len(r))+1)/len(r),np.sort(r), np.sort(r)**2/(1+np.sort(r))**2)
plt.show()
x = np.c_[r*np.cos(phi)*np.sin(theta), r*np.sin(phi)*np.sin(theta), r*np.cos(theta)]

phi = -1/(1+r)
#print(np.average(phi)/2)
v_e = (-2*phi)**0.5

# do Von Neumann sampling
Fq = lambda rv, r: (2*r*(1 + r)**4*rv**2*((np.sqrt(-((-1 + rv**2)*(r + rv**2)))*(-1 + r + 2*rv**2)*(-3 - 14*r - 3*r**2 + 8*(-1 + r)*rv**2 + 8*rv**4))/(1 + r)**4 + 3*np.arcsin(np.sqrt((1 - rv**2)/(1 + r)))))/(np.pi*(r + rv**2)**2.5) #distribution of velocity as a fraction of escape velocity, at a given radius r
Qs = []

@jit
def VonNeumann():
    Qs = []
    for radius in r: 
        while True:
            if radius < 0.1:
                Y = 200*np.random.rand() # 100 is an approximate upper bound
            else:
                Y = 10 * np.random.rand()
            X = np.random.rand()            
            if Y < Fq(X, radius):
                Qs.append(X)
                break
    return np.array(Qs)

v = v_e * VonNeumann()

print(np.sum(v**2 / 2/ N), np.sum(phi)/2/N)
phi_v = np.random.rand(N) * 2 * np.pi
theta_v = np.arccos(2.0*np.random.rand(N)-1.0)

v = np.c_[v*np.cos(phi_v)*np.sin(theta_v), v*np.sin(phi_v)*np.sin(theta_v), v*np.cos(theta_v)]
boxsize = 10.
F = h5py.File("hernquist_%d_m%g_a%g.hdf5"%(int(N**(1./3)+0.5),m, a),'w')
F.create_group("PartType1")
F.create_group("Header")
F["Header"].attrs["NumPart_ThisFile"] = [0,len(x),0,0,0,0]
F["Header"].attrs["MassTable"] = [0,float(m)/N,0,0,0,0]
F["Header"].attrs["BoxSize"] = boxsize
F["Header"].attrs["Time"] = 0.0
F["PartType1"].create_dataset("Coordinates", data=x*a + boxsize/2)
F["PartType1"].create_dataset("Potential", data=phi*float(m)/a)
F["PartType1"].create_dataset("Velocities", data=v*(float(m)/a)**0.5 * G**0.5)
F["PartType1"].create_dataset("Masses", data=np.repeat(float(m)/N,N))
F["PartType1"].create_dataset("ParticleIDs", data=np.arange(N))
F.close()
