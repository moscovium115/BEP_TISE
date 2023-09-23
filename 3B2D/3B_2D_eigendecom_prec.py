#this script is made to reproduce the 2D results from the "tensor products" paper
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

####VERANDER DIT LATER, MASSA RATIO STAAT NU OP 1
M,m=1,1


mass_ratio=M/m
alpha_x=-1/((1+mass_ratio))
alpha_y=(1+2*mass_ratio)/(-4*(1+1*mass_ratio))
print("alpha_x:", alpha_x)
print("alpha_y:", alpha_y)

energy_level=1

#Select which energy level you want to compute
if energy_level==1:
    E_target=1e-1
    v0=0.947343916
elif energy_level==2:
    E_target=1e-2
    v0=0.48272728
else:
    E_target=1e-3
    v0=0.31340752



#N_x boven 40 geeft lagspikes
N_x=20
N_y=int(N_x/2)

print("storage hamiltonian matrix:", 8*(N_x**4 * N_y**4 )/1e9, "GB")

N_x_1=N_x
N_x_2=N_x
N_y_1=N_y
N_y_2=N_y

cheb_nodes_x=np.cos(np.pi*(2*np.arange(1,N_x+1)-1)/(2*N_x))
cheb_nodes_y=np.cos(np.pi*(2*np.arange(1,N_y+1)-1)/(2*N_y))
cheb_nodes_x_original=np.cos(np.pi*(2*np.arange(1,2*N_x_1+1)-1)/(4*N_x_1))
cheb_nodes_y_original=np.cos(np.pi*(2*np.arange(1,2*N_x_1+1)-1)/(4*N_x_1))

#constructing the derivative matrices
Dirac_1_x=construct_first_derivative_matrix(cheb_nodes_x,N_x_1)
Dirac_2_x=construct_second_derivative_matrix(cheb_nodes_x,N_x_1)


Dirac_1_y=construct_first_derivative_matrix(cheb_nodes_y,N_y_1)
Dirac_2_y=construct_second_derivative_matrix(cheb_nodes_y,N_y_1)



L_x=1*0.5*(1/np.sqrt(2*E_target))*(2+1/mass_ratio)/np.sqrt(1+1/mass_ratio)
L_y=1*0.25*(1/np.sqrt(2*E_target))*(2+1/mass_ratio)/np.sqrt(1+1/mass_ratio)
A_x=np.diag((1/L_x)*(1-cheb_nodes_x**2)**(3/2))
B_x=np.diag((-3/L_x**2)*cheb_nodes_x*(1-cheb_nodes_x**2)**(2))

A_y=np.diag((1/L_y)*(1-cheb_nodes_y**2)**(3/2))
B_y=np.diag((-3/L_y**2)*cheb_nodes_y*(1-cheb_nodes_y**2)**(2))

mapping_x=L_x*cheb_nodes_x/np.sqrt((1-cheb_nodes_x**2))
mapping_y=L_y*cheb_nodes_y/np.sqrt((1-cheb_nodes_y**2))

#vermijd A_x @A_x en stel meteen de A matrix op met de kwadraten
D_2_realline_x=A_x@A_x@Dirac_2_x+B_x@Dirac_1_x
#Waarom wordt de selection rule niet meer gebruikt in het 2D geval


D_2_realline_y=A_y@A_y@Dirac_2_y+B_y@Dirac_1_y


identity_x_1_D=np.eye(N_x_1)
identity_x_2_D=np.eye(N_x_2**2)
identity_y_1_D=np.eye(N_y_1)
identity_y_2_D=np.eye(N_y_2**2)


D_x_1=D_2_realline_x

mat_1=sp.sparse.kron(D_2_realline_x,identity_x_1_D)+sp.sparse.kron(identity_x_1_D,D_2_realline_x)
mat_2=sp.sparse.kron(D_2_realline_y,identity_y_1_D)+sp.sparse.kron(identity_y_1_D,D_2_realline_y)
Hamiltonian_TISE=alpha_x*sp.sparse.kron(mat_1,identity_y_2_D)+alpha_y*sp.sparse.kron(identity_x_2_D,mat_2)


V_arr=np.zeros(N_x_1*N_x_2*N_y_1*N_y_2)
V_arr_unperturbed=np.zeros(N_x_1*N_x_2*N_y_1*N_y_2)
V_arr_perturbed=np.zeros(N_x_1*N_x_2*N_y_1*N_y_2)

#setting up the potential matrix
#it is equal to the matlab implementation

#TODO: For loop vectoriseren


#in de potentiaal is er nu interactie tussen deeltje 1 en 2, en dat was er eerder niet
for i in range(N_x_1):
    for j in range(N_x_2):
        for k in range(N_y_1):
            for l in range(N_y_2):
                # val_b=(0.5*mapping_x[i]+mapping_y[j])**2
                # val_c=(0.5*mapping_x[i]-mapping_y[j])**2
                index_num=i*N_x_2*N_y_1*N_y_2+j*N_y_1*N_y_2+k*N_y_2+l
                val_a=mapping_x[i]**2+mapping_x[j]**2
                val_b=0.25*mapping_x[i]**2+0.25*mapping_x[j]**2+mapping_y[k]**2+mapping_y[l]**2+mapping_x[i]*mapping_y[k]+mapping_x[j]*mapping_y[l]
                val_c=0.25*mapping_x[i]**2+0.25*mapping_x[j]**2+mapping_y[k]**2+mapping_y[l]**2-mapping_x[i]*mapping_y[k]-mapping_x[j]*mapping_y[l]
                # print("debug val b:", val_b)
                V_arr[index_num]= -v0*(np.exp(-val_b)+np.exp(-val_c)+0*np.exp(-val_a))
                # V_arr_unperturbed[index_num]= -v0*(np.exp(-val_b))
                # V_arr_perturbed[index_num]= -v0*(np.exp(-val_c))
                # V_arr_perturbed[index_num]= -v0*(np.exp(-val_c))





# print("potential matrix:", V_arr)
V=sp.sparse.diags(V_arr)



Hamiltonian_TISE+=V



from jadapy import jdqr
from jadapy import Target


target=2.30

sigma_1=-abs(alpha_y)*target/(abs(alpha_x+alpha_y))/2
sigma_2=-abs(alpha_x)*target/(abs(alpha_x+alpha_y))/2


A=alpha_x*D_2_realline_x/E_target-sigma_2*identity_x_1_D
B=A
C=alpha_y*D_2_realline_y/E_target-sigma_1*identity_y_1_D
D=C
E=sparse.kron(C,identity_y_1_D)+sparse.kron(identity_y_1_D,D)
F=sparse.kron(A,identity_x_1_D)+sparse.kron(identity_x_1_D,B)


schur_A=schur(A)
schur_B=schur_A
schur_C=schur(C)
schur_D=schur_C


Q_1=sparse.kron(schur_A[1],schur_A[1])
Q_2=sparse.kron(schur_C[1],schur_C[1])
S_1=sparse.kron(schur_A[0],identity_x_1_D)+sparse.kron(identity_x_1_D,schur_A[0])
S_2=sparse.kron(schur_C[0],identity_y_1_D)+sparse.kron(identity_y_1_D,schur_C[0])

len_vector=Hamiltonian_TISE.shape[0]

mat_2=(sparse.kron(S_1,identity_y_2_D)+sparse.kron(identity_x_2_D,S_2))

from scipy.sparse.linalg import spsolve_triangular
from efficient_multiple_kronecker_vec_multiplication import *

As=[schur_A[1].T,schur_A[1].T,schur_C[1].T,schur_C[1].T]
Bs=[schur_A[1],schur_A[1],schur_C[1],schur_C[1]]

dimensie_lijst=[N_x_1,N_x_2,N_y_1,N_y_2]

kron_vec_time=[]
sp_solve_time=[]

#Num of times preconditioner is applied
prec_1_num=0


from scipy.linalg import solve_triangular



def prec_eigen_1(x,*args):
    '''Dezelfde preconditioner alleen zijn de kronecker vector producten versneld nu'''
    global prec_1_num
    prec_1_num+=1
    x=np.reshape(x,(len_vector,1))
    start_time=time.time()
    y_1=kron_vec_prod(As,x)
    kron_vec_time.append(time.time()-start_time)
    # y_1=mat_1@x
    start_time=time.time()
    y_2=spsolve_triangular(mat_2,y_1,lower=False)
    sp_solve_time.append(time.time()-start_time)
    y_3=kron_vec_prod(Bs,y_2)
    y_3=np.reshape(y_3,(len_vector,1))
    return y_3



num_tests=1
time_arr=[]

for i in range(num_tests):
    start_time=time.time()
    # eigenval,eigenvec=jdqr.jdqr(Hamiltonian_TISE/E_target,num=1,target=-target,tol=1e-10,return_eigenvectors=True,subspace_dimensions=(20,50),maxit=2500)
    # eigenval,eigenvec=jdqr.jdqr(Hamiltonian_TISE/E_target,num=1,target=-target,tol=1e-10,return_eigenvectors=True,subspace_dimensions=(20,50),maxit=2500,prec=prec_eigen)
    eigenval,eigenvec=jdqr.jdqr(Hamiltonian_TISE/E_target,num=1,target=-target,tol=1e-10,return_eigenvectors=True,subspace_dimensions=(20,50),maxit=2500,prec=prec_eigen_1)
    # eigenval,eigenvec=jdqr.jdqr(Hamiltonian_TISE/E_target,num=1,target=-target,tol=1e-10,return_eigenvectors=True,subspace_dimensions=(20,50),maxit=2500,prec=NKP_PYTHON_lIB)



    # eigenval,eigenvec=jdqr.jdqr(Hamiltonian_TISE/E_target,num=1,target=-target,tol=1e-10,return_eigenvectors=True,subspace_dimensions=(20,50),maxit=2500,prec=prec_NKP)

    end_time=time.time()
    time_arr.append(end_time-start_time)

print("elapsed time:", np.mean(time_arr))

print("kron vec prod time:", np.mean(kron_vec_time))
print("spsolve time:", np.mean(sp_solve_time))
print("prec_1_num:", prec_1_num)







