import numpy as np
import numpy.ctypeslib as ctl
import ctypes

import matplotlib
matplotlib.use('TkAgg')
from Helper_functions import *

import matplotlib.pyplot as plt





libname="N_body.so"
libdir = './'
mylib=ctl.load_library(libname, libdir)


M,m=20,1
mass_ratio=M/m
# alpha_x=-1/((1+mass_ratio))

#number of chebyshev polynomials
N=100
cheb_nodes=np.cos(np.pi*(2*np.arange(1,N+1)-1)/(2*N))


Dirac_1_x=construct_first_derivative_matrix(cheb_nodes,N)
###Initialize the second derivative matrix
Dirac_2_x=construct_second_derivative_matrix(cheb_nodes,N)

L=1

A=np.diag((1/L)*(1-cheb_nodes**2)**(3/2))
B=np.diag((-3/L**2)*cheb_nodes*(1-cheb_nodes**2)**(2))

mapping=L*cheb_nodes/np.sqrt((1-cheb_nodes**2))

alpha=-0.5
# h_bar=1.0545718e-34
# m=9.10938356e-31
# alpha=-h_bar**2/(2*m)
omega=1
# pot_arr=0.5*m*omega**2*mapping**2
# pot_arr=np.exp(-chebyshev_nodes**2)
# print("potentieel:",pot_arr)

#gaussian potential
# V=-0.34459535*np.exp(-mapping**2)

D_2_realline=A@A@Dirac_2_x+B@Dirac_1_x
from scipy.optimize import dual_annealing
E_0=-0.01
eigenval=[]
def find_eigenval(v_0):
    """find correct v_0 such that first bound state is -E_0"""
    v=v_0
    #gaussian potential
    # pot_arr=-v*np.exp(-mapping**4)
    #harmonic potential
    # pot_arr=-v/(mapping**1)
    # pot_arr=v*mapping
    pot_arr=-v*np.exp(-mapping**2)

    global eigenval
    V=np.diag(pot_arr)
    Hamiltonian_TISE=alpha*D_2_realline+V

    #Weights for the linear combinations are calculated by solving the eigenvalue problem
    eigenval, eigenvectors =np.linalg.eig(Hamiltonian_TISE)
    sorted_eigenvalues=np.sort(eigenval)

    print("eigenvalues:", sorted_eigenvalues[0],v)
    print("residual:",np.abs(sorted_eigenvalues[0]-E_0))
    # print("Eigenvalues:", sorted_eigenvalues/sorted_eigenvalues[0])
    # print("Calculated Energy:",np.min(eigenval),np.min(eigenval.real))
    return np.abs(sorted_eigenvalues[0]-E_0)

from scipy.optimize import minimize

#coulomb potential:
#oplossing: 0.07071704488992694
# x_0=0.07071704488992694
#guassian inital guess
# x_0=0.08887512312312

x_0=0.025410150076406388
# x_0=-0.00019718




#hele accurate waarde voor gaussian
# x_0=0.08887372057815747

find_eigenval(x_0)

result = minimize(find_eigenval, x_0, method='Nelder-Mead', options={'maxiter': 100},tol=1e-15)
#
# # find_eigenval(0.34459535,1e-1)
print("oplossing:",result.x[0])
print("all sorted eigenvalues:",np.sort(eigenval))