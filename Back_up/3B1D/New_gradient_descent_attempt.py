#this script is made to reproduce the 1D results from the "tensor products" paper
import numpy as np
import numpy.ctypeslib as ctl
import ctypes
from scipy import sparse
import scipy as sp
import matplotlib
from jadapy import jdqr
from jadapy import Target
from scipy.linalg import solve_sylvester
from scipy.linalg import schur, eigvals
from Helper_functions import *
import time


matplotlib.use('TkAgg')

M,m=20,1
mass_ratio=M/m
alpha_x=-1/((1+mass_ratio))
alpha_y=(1+2*mass_ratio)/(-4*(1+1*mass_ratio))
energy_level=3


#Select which energy level you want to compute
if energy_level==1:
    E_target=1e-1
    v0=0.34459535
elif energy_level==2:
    E_target=1e-2
    v0=0.08887372
else:
    E_target=1e-3
    v0=0.02613437

N_x=30
N_y=int(N_x/2)


cheb_nodes_x=np.cos(np.pi*(2*np.arange(1,N_x+1)-1)/(4*N_x))
cheb_nodes_y=np.cos(np.pi*(2*np.arange(1,N_y+1)-1)/(4*N_y))
cheb_nodes_x_original=np.cos(np.pi*(2*np.arange(1,2*N_x+1)-1)/(4*N_x))
cheb_nodes_y_original=np.cos(np.pi*(2*np.arange(1,2*N_y+1)-1)/(4*N_y))

#constructing the derivative matrices
Dirac_1_x=construct_first_derivative_matrix(cheb_nodes_x_original,2*N_x)
Dirac_1_y=construct_first_derivative_matrix(cheb_nodes_y_original,2*N_y)
Dirac_2_x=construct_second_derivative_matrix(cheb_nodes_x_original,2*N_x)
Dirac_2_y=construct_second_derivative_matrix(cheb_nodes_y_original,2*N_y)

L_x=3*(1/np.sqrt(2*E_target))*(2+1/mass_ratio)/np.sqrt(1+1/mass_ratio)
L_y=1.5*(1/np.sqrt(2*E_target))*(2+1/mass_ratio)/np.sqrt(1+1/mass_ratio)

A_x=np.diag((1/L_x)*(1-cheb_nodes_x_original**2)**(3/2))
A_y=np.diag((1/L_y)*(1-cheb_nodes_y_original**2)**(3/2))
#B matrix
B_x=np.diag((-3/L_x**2)*cheb_nodes_x_original*(1-cheb_nodes_x_original**2)**(2))
B_y=np.diag((-3/L_y**2)*cheb_nodes_y_original*(1-cheb_nodes_y_original**2)**(2))

mapping_x=L_x*cheb_nodes_x/np.sqrt((1-cheb_nodes_x**2))
mapping_y=L_y*cheb_nodes_y/np.sqrt((1-cheb_nodes_y**2))

#vermijd A_x @A_x en stel meteen de A matrix op met de kwadraten
D_2_realline_x=A_x@A_x@Dirac_2_x+B_x@Dirac_1_x
#we maken de D2 matrix kleiner, en bij de tweede term die gesommeerd wordt, vervangen we de kolommen van volgorde
D_2_realline_x=D_2_realline_x[0:N_x,0:N_x]+D_2_realline_x[0:N_x,np.arange(2*N_x-1,N_x-1,step=-1)]

D_2_realline_y=A_y@A_y@Dirac_2_y+B_y@Dirac_1_y
D_2_realline_y=D_2_realline_y[0:N_y,0:N_y]+D_2_realline_y[0:N_y,np.arange(2*N_y-1,N_y-1,step=-1)]

#
Id_y=np.identity(N_y)
Id_x=np.identity(N_x)

Test_x=mapping_x
Test_y=mapping_y

# Convert the result list to a NumPy array
# result_array = np.array(result)
# Perform broadcasting to compute the result
result_array_1 = 0.5*Test_x[:, np.newaxis] +  Test_y[np.newaxis, :]
result_array_2 =0.5* Test_x[:, np.newaxis] - Test_y[np.newaxis, :]

# Print the result
rel_distance_1=np.abs(np.ravel(result_array_1))
rel_distance_2=np.abs(np.ravel(result_array_2))

V_test=-v0*(np.exp(-(rel_distance_1**2))+np.exp(-(rel_distance_2**2)))
V=sp.sparse.diags(V_test)








#kronecker product optimaliseren
Hamiltonian_TISE=alpha_x*sparse.kron(D_2_realline_x,Id_y)+(alpha_y)*sparse.kron(Id_x, D_2_realline_y)+V
target=2.71

sigma_1=-abs(alpha_y)*target/(abs(alpha_x+alpha_y))
sigma_2=-abs(alpha_x)*target/(abs(alpha_x+alpha_y))


#solving sylvester equation
A=(alpha_y*D_2_realline_y)/E_target-sigma_1*Id_y
B=(alpha_x*D_2_realline_x)/E_target-sigma_2*Id_x
# B=B.T



schur_A=schur(A)
schur_B=schur(B)

def pseudo_inverse_diagonal(matrix):
    diagonal = np.array(matrix)
    n = len(diagonal)
    inv_diagonal=1/diagonal
    # print(inv_diagonal)
    inv_diagonal[np.isinf(inv_diagonal)]=1
    pseudo_inv=inv_diagonal
    return sp.sparse.diags(pseudo_inv)

# print("debug 2:", V.diagonal())
# diag_prec_V=pseudo_inverse_diagonal(V.diagonal()/E_target)
# print("debug hier:",diag_prec_V@V/E_target)

# Q=sp.sparse.kron(B,Id_y)+sp.sparse.kron(Id_x,A)
Q=sp.sparse.kron(B,Id_y)+sp.sparse.kron(Id_x,A)+V/E_target

# Q_inv=sp.sparse.linalg.inv(Q)
R_1,Q_1=schur_B
R_2,Q_2=schur_A

Prec_2=np.kron(Q_1.T,Q_2.T)
# print(Prec_2)
Prec_2=np.linalg.inv(np.kron((R_1),np.eye(N_y))+np.kron(np.eye(N_x),(R_2)))@Prec_2

# Prec_2=np.linalg.inv(np.kron((np.diag(R_1.diagonal())),np.eye(N_y))+np.kron(np.eye(N_x),(np.diag(R_2.diagonal()))))@Prec_2
Prec_2=np.kron(Q_1,Q_2)@Prec_2

Q_inv=Prec_2
Q_til=Q_inv@Q

print("Condition number before preconditioning:",np.linalg.cond(Q.toarray()))
print("Condition number after preconditioning:",np.linalg.cond(Q_til))

def prec(x):
    return Q_inv@x

def prec_T(x):
    return Q_inv.T@x


def gd_solver(x):
    x=np.reshape(x,(N_x*N_y,1))
    e_til=prec(x)
    x_k=np.zeros([N_x*N_y,1])
    iterations=10
    #Nu preconditioner in verwerken
    for i in range(iterations):
        residual=e_til-prec(Q@x_k)
        #berekening van tau kan veel efficienter
        tau=np.linalg.norm(Q.T@prec_T(residual),ord="fro")**2/np.linalg.norm(prec(Q@Q.T@prec_T(residual)),ord="fro")**2
        # tau=1
        print("tau=",tau)
        print("residual:",np.linalg.norm(residual))
        x_k=x_k+tau*Q.T@prec_T(residual)
    print("final residual:",np.linalg.norm(residual))
    # x_k=prec(x)
    return x_k

E=np.random.rand(N_x,N_y)

e_vec=np.reshape(E,(N_x*N_y,1))
gd_solver(e_vec)
