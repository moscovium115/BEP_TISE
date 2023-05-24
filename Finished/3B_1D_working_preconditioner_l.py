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


N_y=100
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
A=alpha_y*D_2_realline_y/E_target-sigma_1*np.identity(N_y)
B=alpha_x*D_2_realline_x/E_target-sigma_2*np.identity(N_x)
B=B.T
A=np.asarray(A, dtype=np.float128)
B=np.asarray(B, dtype=np.float128)

schur_A=schur(A)
schur_B=schur(B)

def prec_func(x, *args):
    #function approximates w in (A-sigma*I)w=x
    C=np.reshape(x,(N_x,N_y))
    C=np.asarray(C, dtype=np.float128)
    X = solve_sylvester(A,B, C.T,schur_A,schur_B)
    result_test=np.reshape(X.T,(N_x*N_y,1))
    return result_test

num_tests=1

test_arr=[]
print("dtypeee hamil:", Hamiltonian_TISE.dtype)
for test in range(num_tests):
    start_time = time.time()
    eigenval,eigenvec=jdqr.jdqr(Hamiltonian_TISE/E_target,num=1,target=-2.71,tol=1e-10,return_eigenvectors=True,subspace_dimensions=(20,50),maxit=2500,prec=prec_func)
    # eigenval,eigenvec=jdqr.jdqr(Hamiltonian_TISE/E_target,num=1,target=-2.71,tol=1e-10,return_eigenvectors=True,subspace_dimensions=(20,50),maxit=2500)
    end_time=time.time()
    test_arr.append(end_time-start_time)

avg_time_1=np.mean(test_arr)
std_time_1=np.std(test_arr)
print("average execution time:",avg_time_1)
print("Found eigenvalue:",np.real(eigenval[0]))
print("all eigenvalues:",np.real(eigenval))
print(len(eigenvec))
num_eigenvec=0
print(eigenvec[5,num_eigenvec],eigenvec[5+N_y+1,num_eigenvec])
print(eigenvec)

#Dit in jadapy uitproberen
#De hamilton matrix class maken met overrided matrix multiplication symbol, gebruik chatgpt hiervoor

# start_vec=np.ones((N_x*N_y,1))
# start_vec.reshape((N_x*N_y,1))
test_vec=np.arange(A.shape[0]*B.shape[0])
test_vec.reshape((test_vec.shape[0],1))

def normal_product(test_vec):
    res_1=Hamiltonian_TISE@test_vec
    res_1=np.reshape(res_1,(res_1.shape[0],1))
    return res_1

#hieronder werkt het wel
A=alpha_x*D_2_realline_x
B=alpha_y*D_2_realline_y

res_1=normal_product(test_vec)
def improved_product(test_vec):
    W=np.reshape(test_vec,(N_x,N_y)).T
    test_vec_2=B@W+W@A.T
    test_vec=test_vec.reshape((N_x*N_y,1))
    test_vec_2=np.reshape(test_vec_2.T,(N_x*N_y,1))+V@test_vec
    return test_vec_2

import numpy

#Misschien gaat er iets mis met de letters ofzooo???? ik snap het niet
#verander A naar gewoon A_mat zodat er geen verwarring in jadapy kan optreden



#nu nog E target erinverwerken
#hij rekent ook echt veel sneller uit!! super nice



time_1=time.time()
res_5=improved_product(test_vec)/E_target
time_2=time.time()
fast_speed=time_2-time_1
print("res 5:",res_5.dtype)
print("testtt timeee yess:",fast_speed)

time_1=time.time()
res=(Hamiltonian_TISE@test_vec)/E_target
# res=normal_product(test_vec)/E_target
res=np.reshape(res,(res_1.shape[0],1))

time_2=time.time()
norm_speed=time_2-time_1
print("normal speed:",norm_speed)
print("verschil in snelheid:",100*(norm_speed-fast_speed)/fast_speed,"%")
print("foutttjee verschil:", 100*(res_5-res)/res)


from scipy.sparse.linalg import LinearOperator

matrix_test_mv=A.T
def mv(v):
    W=np.reshape(v,(N_x,N_y)).T
    test_vec_2=B@W+W@A.T
    test_vec=v.reshape((N_x*N_y,1))
    test_vec_2=np.reshape(test_vec_2.T,(N_x*N_y,1))+V@test_vec
    return test_vec_2/E_target
    # return improved_product(v)/E_target
    # return normal_product(v)/E_target

test_mat = LinearOperator((N_x*N_y,N_x*N_y), matvec=mv)

#vgm gaat er iets mis bij LinearOperator, want als we return hebben de normal product zien we dat dat veeeel langer duurt dan zonder lineaire operator
#we krijgen wel performance gains bij grote waarden van N_x, maar dit hoort volgens mij niet
test_arr=[]
for i in range(num_tests):
    start_time = time.time()
    eigenval,eigenvec=jdqr.jdqr(test_mat,num=1,target=-2.71,tol=1e-10,return_eigenvectors=True,subspace_dimensions=(20,50),maxit=2500,prec=prec_func)
    end_time=time.time()
    test_arr.append(end_time-start_time)

avg_time_2=np.mean(test_arr)
std_time_2=np.std(test_arr)

print("average execution time linear operator:",avg_time_2,"std:",std_time_2)
print("average execution time without linear operator:",avg_time_1,"std:",std_time_1)