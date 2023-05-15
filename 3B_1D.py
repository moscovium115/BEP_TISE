import numpy as np
from scipy import sparse
import scipy as sp

import numpy as np
import numpy.ctypeslib as ctl
import ctypes

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
libname="N_body.so"
libdir = './'
mylib=ctl.load_library(libname, libdir)

#number of chebyshev polynomials
N=50
cheb_nodes=np.cos(np.pi*(2*np.arange(1,N+1)-1)/(2*N))


###Initialize the first derivative matrix
Initialize_First_Derivative_function=mylib.Init_First_derivative_matrix
Initialize_First_Derivative_function.restype = ctypes.c_longlong
Initialize_First_Derivative_function.argtypes = [ctl.ndpointer(np.float64,flags='aligned, c_contiguous'),ctl.ndpointer(np.float64,flags='aligned, c_contiguous'),ctypes.c_int]
Dirac_1=np.ones((N,N))
#changes Dirac_1 in place
Initialize_First_Derivative_function(Dirac_1,cheb_nodes,N)


###Initialize the second derivative matrix
Initialize_Second_Derivative_function=mylib.Init_Second_derivative_matrix
Initialize_Second_Derivative_function.restype = ctypes.c_longlong
Initialize_Second_Derivative_function.argtypes = [ctl.ndpointer(np.float64,flags='aligned, c_contiguous'),ctl.ndpointer(np.float64,flags='aligned, c_contiguous'),ctypes.c_int]

Dirac_2=np.ones((N,N))
Initialize_Second_Derivative_function(Dirac_2,cheb_nodes,N)
print(Dirac_2)



L=1

A=np.diag((1/L)*(1-cheb_nodes**2)**(3/2))
#B matrix
B=np.diag((-3/L**2)*cheb_nodes*(1-cheb_nodes**2)**(2))

mapping=L*cheb_nodes/np.sqrt((1-cheb_nodes**2))

# alpha=-0.5
h_bar=1.0545718e-34
m=9.10938356e-31
alpha=-h_bar**2/(2*m)
omega=0.01
pot_arr=0.5*m*omega**2*mapping**2
# pot_arr=np.exp(-chebyshev_nodes**2)
print("potentieel:",pot_arr)
V=np.diag(pot_arr)

D_2_realline=A@A@Dirac_2+B@Dirac_1



# Hamiltonian_TISE=alpha*D_2_realline+V

alpha_x=1
alpha_y=1
Id_y=np.identity(N)
print(Id_y)
Hamiltonian_TISE=-(alpha_x/2)*sparse.kron(D_2_realline,Id_y)-(alpha_y/2)*sparse.kron(Id_y, D_2_realline)
print(Hamiltonian_TISE)

from jadapy import jdqr
from jadapy import Target


inv=sparse.kron((-alpha_x/2)*D_2_realline,np.identity(N))
print(type(inv))
# inv=sp.in

def _prec(x, *args):
    return inv.dot(x)
print(np.linalg.det(Hamiltonian_TISE.toarray()))
jdqr.jdqr(Hamiltonian_TISE, num=1,tol=1e-2,target=Target.SmallestMagnitude,prec=_prec)
# jdqr.jdqr(Hamiltonian_TISE, num=1, subspace_dimensions=(30,40),target=Target.SmallestMagnitude)


