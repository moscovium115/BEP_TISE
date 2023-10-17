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

N_x=64
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

#guassian potential
V_test=-v0*(np.exp(-(rel_distance_1**2))+np.exp(-(rel_distance_2**2)))
# v0=-2.4678047338966283e-08
# V_test=-v0*((rel_distance_1**2)+(rel_distance_2**2))

V=sp.sparse.diags(V_test)






A_1=alpha_x*D_2_realline_x
B_1=alpha_y*D_2_realline_y
from scipy.sparse.linalg import LinearOperator

mat_vec_count=0

def mv(v):
    W=np.reshape(v,(N_x,N_y)).T
    test_vec_2=B_1@W+W@A_1.T
    test_vec=v.reshape((N_x*N_y,1))
    test_vec_2=np.reshape(test_vec_2.T,(N_x*N_y,1))+V@test_vec
    global mat_vec_count
    mat_vec_count+=1
    return test_vec_2

#kronecker product optimaliseren
Hamiltonian_TISE= LinearOperator((N_x*N_y,N_x*N_y), matvec=mv)

target=2.71

def Q_mv(v):
    return (Hamiltonian_TISE@v)/E_target+target*v

Q=LinearOperator((N_x*N_y,N_x*N_y), matvec=Q_mv)


def Q_T_mv(v):
    # print("debug:",v.shape)
    v=v.reshape((N_x*N_y,1))
    W=np.reshape(v,(N_x,N_y)).T
    test_vec_2=B_1.T@W+W@A_1
    test_vec=v.reshape((N_x*N_y,1))
    test_vec_2=np.reshape(test_vec_2.T,(N_x*N_y,1))+V@test_vec
    result_veccie=test_vec_2/E_target+target*v
    result_veccie=np.reshape(result_veccie,(N_x*N_y,1))
    return result_veccie


Q_T=LinearOperator((N_x*N_y,N_x*N_y), matvec=Q_T_mv)



######

sigma_1=-abs(alpha_y)*target/(abs(alpha_x+alpha_y))
sigma_2=-abs(alpha_x)*target/(abs(alpha_x+alpha_y))


#solving sylvester equation
A=(alpha_y*D_2_realline_y)/E_target-sigma_1*Id_y
B=(alpha_x*D_2_realline_x)/E_target-sigma_2*Id_x
# B=B.T


schur_A=schur(A)
schur_B=schur(B)



#uncomment voor bartel stewart preconditioner
# def prec(x,*args):
#     #function approximates w in (A-sigma*I)w=x
#     # print("bruh 1")
#
#     C=np.reshape(x,(N_x,N_y))
#     X = solve_sylvester(B,A,C,schur_B,schur_A)
#     #lost AX+XB^T =C op
#     # print("debug norm:",np.linalg.norm(A@X+X@B.T-C))
#     result_test=np.reshape(X,(N_x*N_y,1))
#     # print("bruh 2")
#     return result_test
#
# def prec_T(x):
#     # print("bruh_1")
#     C=np.reshape(x,(N_x,N_y))
#     X = solve_sylvester(B.T,A.T,C,schur_B,schur_A)
#     result_test=np.reshape(X,(N_x*N_y,1))
#     # print("bruh2")
#     return result_test

R_1,Q_1=schur_B
R_2,Q_2=schur_A
prec_matrix=sp.sparse.kron(R_1,Id_y)+sp.sparse.kron(Id_x,R_2)
prec_matrix=sp.sparse.diags(1/prec_matrix.diagonal())


def prec(x,*args):
    "preconditioner for Q matrix"
    # print(Q.diagonal())
    # print(a)
    x_1=kron_vec_prod([Q_1.T,Q_2.T],x)
    prec_x=prec_matrix@x_1
    prec_x=kron_vec_prod([Q_1,Q_2],prec_x)
    prec_x=np.reshape(prec_x,(N_x*N_y,1))
    return prec_x

def prec_T(x,*args):
    "preconditioner for Q transposed"
    x_1=kron_vec_prod([Q_1.T,Q_2.T],x)
    prec_x=prec_matrix@x_1
    prec_x=kron_vec_prod([Q_1,Q_2],prec_x)
    prec_x=np.reshape(prec_x,(N_x*N_y,1))
    return prec_x

prec_calls=0
# print("potential matrix:",sp.sparse.linalg.norm(V))
# print("amount of zeros potential matrix:",(V.nnz)/V.shape[0])
def geometric_prec(x,*args):
    x=np.reshape(x,(N_x*N_y,1))
    e_til=prec(x)
    x_k=np.zeros([N_x*N_y,1])
    iterations=10
    #Nu preconditioner in verwerken
    for i in range(iterations):
        residual=e_til-prec(Q@x_k)
        #berekening van tau kan veel efficienter
        # tau=np.linalg.norm(Q_T@prec_T(residual),ord="fro")**2/np.linalg.norm(prec(Q@Q_T@prec_T(residual)),ord="fro")**2
        tau=1

        # print("tau=",tau)
        # print("residual:",np.linalg.norm(residual))
        x_k=x_k+tau*Q_T@prec_T(residual)
        # res=np.linalg.norm(residual)
        # if res<1e-15:
            # print(i)
            # break
    # print("final residual:",np.linalg.norm(residual))
    global prec_calls
    prec_calls+=1
    # x_k=e_til
    # print(i,res)
    return x_k



from efficient_multiple_kronecker_vec_multiplication import *


num_tests=1
test_arr=[]
for test in range(num_tests):
    start_time = time.time()
    # eigenval,eigenvec=jdqr.jdqr(Hamiltonian_TISE/E_target,num=1,target=-2.71,tol=1e-10,return_eigenvectors=True,subspace_dimensions=(20,50),maxit=2500,prec=prec_func)
    # eigenval,eigenvec=jdqr.jdqr(Hamiltonian_TISE/E_target,num=1,target=-2.71,tol=1e-10,return_eigenvectors=True,subspace_dimensions=(20,50),maxit=2500)
    # eigenval,eigenvec=jdqr.jdqr(Hamiltonian_TISE/E_target,num=1,target=Target.SmallestRealPart,tol=1e-10, return_eigenvectors=True, arithmetic="complex", subspace_dimensions=(5,50), maxit=2500,prec=geometric_prec)
    eigenval,eigenvec=jdqr.jdqr(Hamiltonian_TISE/E_target,num=1,target=Target.SmallestRealPart,tol=1e-10,return_eigenvectors=True,arithmetic="complex",subspace_dimensions=(20,50),maxit=2500,prec=geometric_prec)
    end_time=time.time()
    test_arr.append(end_time-start_time)

print(np.mean(test_arr))

print("number preconditioner calls:", prec_calls)