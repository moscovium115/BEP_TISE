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
N=1000
cheb_nodes=np.cos(np.pi*(2*np.arange(1,N+1)-1)/(2*N))


Dirac_1_x=construct_first_derivative_matrix(cheb_nodes,N)
###Initialize the second derivative matrix
Dirac_2_x=construct_second_derivative_matrix(cheb_nodes,N)

L=1

A=np.diag((1/L)*(1-cheb_nodes**2)**(3/2))
B=np.diag((-3/L**2)*cheb_nodes*(1-cheb_nodes**2)**(2))

mapping=L*cheb_nodes/np.sqrt((1-cheb_nodes**2))

# alpha=-0.5
h_bar=1.0545718e-34
m=9.10938356e-31
alpha=-h_bar**2/(2*m)
omega=1
pot_arr=0.5*m*omega**2*mapping**2
# pot_arr=np.exp(-chebyshev_nodes**2)
print("potentieel:",pot_arr)
V=np.diag(pot_arr)

#gaussian potential
# V=-0.34459535*np.exp(-mapping**2)

D_2_realline=A@A@Dirac_2_x+B@Dirac_1_x

Hamiltonian_TISE=alpha*D_2_realline+V

#Weights for the linear combinations are calculated by solving the eigenvalue problem
eigenval, eigenvectors =np.linalg.eig(Hamiltonian_TISE)
sorted_eigenvalues=np.sort(eigenval)

print("Eigenvalues:", sorted_eigenvalues/sorted_eigenvalues[0])
print("Calculated Energy:",np.min(eigenval),np.min(eigenval.real))


index_real_eigenval=np.where(eigenval==(eigenval.real))
print("index:",index_real_eigenval[0])


Mode_eigenvector=0
indices_sorted_eigenvalues=np.where(eigenval==sorted_eigenvalues[Mode_eigenvector])
Num_eigenvector=indices_sorted_eigenvalues[0][0]
weights_TISE=eigenvectors[:,Num_eigenvector]

def save_eig_to_txt():
    """Saves all eigenvalues to a txt file"""
    f = open("eigenvalues.txt", "w")
    # sorted_eigenvalues=np.sort(eigenval)
    for i in range(len(eigenval)):
        f.write(str(sorted_eigenvalues[i])+"\n")
    f.close()

def save_eigvec_to_txt():
    """Saves only a selected eigenvector to a txt file"""
    f = open("eigenvectors.txt", "w")
    for i in range(len(weights_TISE)):
        f.write(str(weights_TISE[i])+"\n")
    f.close()


save_eig_to_txt()
save_eigvec_to_txt()

print(weights_TISE)
start_x=-1.0
end_x=1.0
