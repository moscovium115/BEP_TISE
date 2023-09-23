#this script is made to reproduce the 1D results from the "tensor products" paper
#Deze script is super memory efficent, kan N_x=1000 aan op een laptop zonder probleem

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
energy_level=2


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

N_y=50
N_x=int(N_y/2)


cheb_nodes_x=np.cos(np.pi*(2*np.arange(1,N_x+1)-1)/(4*N_x))
cheb_nodes_y=np.cos(np.pi*(2*np.arange(1,N_y+1)-1)/(4*N_y))
cheb_nodes_x_original=np.cos(np.pi*(2*np.arange(1,2*N_x+1)-1)/(4*N_x))
cheb_nodes_y_original=np.cos(np.pi*(2*np.arange(1,2*N_y+1)-1)/(4*N_y))

#constructing the derivative matrices
Dirac_1_x=construct_first_derivative_matrix(cheb_nodes_x_original,2*N_x)
Dirac_1_y=construct_first_derivative_matrix(cheb_nodes_y_original,2*N_y)
Dirac_2_x=construct_second_derivative_matrix(cheb_nodes_x_original,2*N_x)
Dirac_2_y=construct_second_derivative_matrix(cheb_nodes_y_original,2*N_y)

L_y=3*(1/np.sqrt(2*E_target))*(2+1/mass_ratio)/np.sqrt(1+1/mass_ratio)
L_x=1.5*(1/np.sqrt(2*E_target))*(2+1/mass_ratio)/np.sqrt(1+1/mass_ratio)

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
del cheb_nodes_x
del cheb_nodes_y
del cheb_nodes_x_original
del cheb_nodes_y_original
del A_x
del A_y
del B_x
del B_y
del Dirac_1_x
del Dirac_1_y
del Dirac_2_x
del Dirac_2_y
del L_x
del L_y

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
# Hamiltonian_TISE=alpha_x*sparse.kron(D_2_realline_x,Id_y)+(alpha_y)*sparse.kron(Id_x, D_2_realline_y)+V

A_1=alpha_x*D_2_realline_x
B_1=alpha_y*D_2_realline_y
from scipy.sparse.linalg import LinearOperator


def mv(v):
    W=np.reshape(v,(N_x,N_y)).T
    test_vec_2=B_1@W+W@A_1.T
    test_vec=v.reshape((N_x*N_y,1))
    test_vec_2=np.reshape(test_vec_2.T,(N_x*N_y,1))+V@test_vec
    return test_vec_2
    # return improved_product(v)/E_target
    # return normal_product(v)/E_target

Hamiltonian_TISE= LinearOperator((N_x*N_y,N_x*N_y), matvec=mv)
target=2.71

sigma_1=-abs(alpha_y)*target/(abs(alpha_x+alpha_y))
sigma_2=-abs(alpha_x)*target/(abs(alpha_x+alpha_y))


#solving sylvester equation
A=(alpha_x*D_2_realline_x)/E_target-sigma_2*Id_x
B=(alpha_y*D_2_realline_y)/E_target-sigma_1*Id_y
# A=np.asarray(A, dtype=np.float64)
# B=np.asarray(B, dtype=np.float64)

schur_A=schur(A)
schur_B=schur(B)

prec_arr=sp.sparse.kron(sp.sparse.diags(schur_A[0].diagonal()),Id_y)+sp.sparse.kron(Id_x,sp.sparse.diags(schur_B[0].diagonal()))
prec_arr=prec_arr.diagonal()
diag_prec=sp.sparse.diags(1/prec_arr)

#weet niet of dit klopt
As=[schur_A[1].T,schur_B[1].T]
Bs=[schur_A[1],schur_B[1]]

from efficient_multiple_kronecker_vec_multiplication import *

kron_vec_arr=[]
from memory_profiler import profile
# @profile
def prec_func(x, *args):
    #function approximates w in (A-sigma*I)w=x
    x=np.reshape(x,(N_x*N_y,1))
    time_1=time.time()
    y_1=kron_vec_prod(As,x)
    kron_vec_arr.append(time.time()-time_1)
    y_1=diag_prec@y_1
    y_2=kron_vec_prod(Bs,y_1)
    y_2=y_2.reshape((N_x*N_y,1))
    return y_2

def prec_func_bartel(x, *args):
    #function approximates w in (A-sigma*I)w=x
    C=np.reshape(x,(N_x,N_y))
    C=np.asarray(C, dtype=np.float64)
    X = solve_sylvester(A,B, C.T,schur_A,schur_B)
    result_test=np.reshape(X.T,(N_x*N_y,1))
    return result_test


num_tests=1

test_arr=[]
for test in range(num_tests):
    start_time = time.time()
    eigenval,eigenvec=jdqr.jdqr(Hamiltonian_TISE/E_target,num=1,target=-2.71,tol=1e-10,return_eigenvectors=True,subspace_dimensions=(20,50),maxit=2500,prec=prec_func)
    # eigenval,eigenvec=jdqr.jdqr(Hamiltonian_TISE/E_target,num=1,target=-2.71,tol=1e-10,return_eigenvectors=True,subspace_dimensions=(20,50),maxit=2500,prec=prec_func_bartel)
    # eigenval,eigenvec=jdqr.jdqr(Hamiltonian_TISE/E_target,num=1,target=-2.71,tol=1e-10,return_eigenvectors=True,subspace_dimensions=(20,50),maxit=2500)
    end_time=time.time()
    test_arr.append(end_time-start_time)

avg_time_1=np.mean(test_arr)
std_time_1=np.std(test_arr)
print("avg time:", avg_time_1)

#no prec 0.42 28 iteraties
#prec 0.95 seconden 28 iteraties

#N_x=10, N_y=20
#0.096 s 10 samples no prec 27 iteraties
#0.075 s 10 samples prec 20 iteraties

#N_x=20, N_y=40 10 samples
#avg time: 0.285 s no prec 28 iteraties
#avg time: 0.350 s prec 24 iteraties

#N_x=50, N_y=100 10 samples
#bartel stewart: avg time: 0.47 s prec 23 iteraties
#alternate prec: 0.138 s prec 24 iteraties



#N_x=50, N_y=100 sample size=100:

#bartel stewart prec:
#avg time: 0.47129723787307737, 23 iterations

#alternate preconditioner:
#avg time: 0.4399633479118347, 24 iterations

#no prec: 1.563332326412201, 29 iterations
print("kron vec prod time:", np.mean(kron_vec_arr))

from guppy import hpy
h = hpy()
print(h.heap())