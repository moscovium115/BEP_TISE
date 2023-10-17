#this script is made to reproduce the 2D results from the "tensor products" paper
import numpy as np
import numpy.ctypeslib as ctl
import ctypes
from scipy import sparse
import scipy as sp
import matplotlib
from jadapy import jdqr
from jadapy import Target
from scipy.sparse.linalg import spsolve_triangular
from efficient_multiple_kronecker_vec_multiplication import *
from scipy.linalg import solve_triangular
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import schur, eigvals
from Helper_functions import *
import time

# matplotlib.use('TkAgg')

####VERANDER DIT LATER, MASSA RATIO STAAT NU OP 1
M,m=1,1


mass_ratio=M/m
alpha_x=-1/((1+mass_ratio))
alpha_y=(1+2*mass_ratio)/(-4*(1+1*mass_ratio))
print("alpha_x:", alpha_x)
print("alpha_y:", alpha_y)

energy_level=2

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
N_x=64
N_y=int(N_x/2)

print("storage hamiltonian matrix:", 8*(N_x**4 * N_y**4 )/1e9, "GB")

N_x_1=N_x
N_x_2=N_x
N_y_1=N_y
N_y_2=N_y

#de chebyshev nodes zijn nu correct dat waren ze eerst niet, check met matlab code nog na of de matrices en alles klopt nu.
#verwerk dit na de check in de andere programmas
cheb_nodes_x=np.cos(np.pi*(2*np.arange(1,N_x+1)-1)/(2*N_x))
cheb_nodes_y=np.cos(np.pi*(2*np.arange(1,N_y+1)-1)/(2*N_y))
cheb_nodes_x_original=np.cos(np.pi*(2*np.arange(1,2*N_x+1)-1)/(4*N_x))
cheb_nodes_y_original=np.cos(np.pi*(2*np.arange(1,2*N_y+1)-1)/(4*N_y))

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


identity_x_1_D=sp.sparse.eye(N_x_1)
identity_x_2_D=sp.sparse.eye(N_x_2**2)
identity_y_1_D=sp.sparse.eye(N_y_1)
identity_y_2_D=sp.sparse.eye(N_y_2**2)


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

# mat_1=sp.sparse.kron(D_2_realline_x,identity_x_1_D)+sp.sparse.kron(identity_x_1_D,D_2_realline_x)
# mat_2=sp.sparse.kron(D_2_realline_y,identity_y_1_D)+sp.sparse.kron(identity_y_1_D,D_2_realline_y)
# Hamiltonian_TISE=alpha_x*sp.sparse.kron(mat_1,identity_y_2_D)+alpha_y*sp.sparse.kron(identity_x_2_D,mat_2)


# V_arr=np.zeros(N_x_1*N_x_2*N_y_1*N_y_2)
# V_arr_unperturbed=np.zeros(N_x_1*N_x_2*N_y_1*N_y_2)
# V_arr_perturbed=np.zeros(N_x_1*N_x_2*N_y_1*N_y_2)

#setting up the potential matrix
#it is equal to the matlab implementation

#TODO: For loop vectoriseren



# Assuming you have the following values defined:
# N_x_1, N_x_2, N_y_1, N_y_2, mapping_x, mapping_y, v0, V_arr

# Create indices arrays for i, j, k, l
i = np.arange(N_x_1)
j = np.arange(N_x_2)
k = np.arange(N_y_1)
l = np.arange(N_y_2)

# Create index grids for all combinations of i, j, k, l
I, J, K, L = np.meshgrid(i, j, k, l, indexing='ij')
del i
del j
del k
del l
# Compute the corresponding index_num
# index_num = I * (N_x_2 * N_y_1 * N_y_2) + J * (N_y_1 * N_y_2) + K * N_y_2 + L

# Compute val_a, val_b, and val_c using broadcasting
mapping_x_i = mapping_x[I]
mapping_x_j = mapping_x[J]
mapping_y_k = mapping_y[K]
mapping_y_l = mapping_y[L]
del I
del J
del K
del L

val_a = mapping_x_i**2 + mapping_x_j**2
val_b = 0.25 * mapping_x_i**2 + 0.25 * mapping_x_j**2 + mapping_y_k**2 + mapping_y_l**2 + mapping_x_i * mapping_y_k + mapping_x_j * mapping_y_l
val_c = 0.25 * mapping_x_i**2 + 0.25 * mapping_x_j**2 + mapping_y_k**2 + mapping_y_l**2 - mapping_x_i * mapping_y_k - mapping_x_j * mapping_y_l
del mapping_x_i
del mapping_x_j
del mapping_y_k
del mapping_y_l
# Compute V_arr using vectorized operations
V_arr = -v0 * (np.exp(-val_b) + np.exp(-val_c) + 0 * np.exp(-val_a))
del val_a
del val_b
del val_c



V_arr=V_arr.ravel()

print(np.min(V_arr))

# print("potential matrix:", V_arr)
V=sp.sparse.diags(V_arr)
del V_arr



target=2.30
sigma_1=-abs(alpha_y)*target/(abs(alpha_x+alpha_y))/2
sigma_2=-abs(alpha_x)*target/(abs(alpha_x+alpha_y))/2


A=alpha_x*D_2_realline_x/E_target-sigma_2*identity_x_1_D
B=A
C=alpha_y*D_2_realline_y/E_target-sigma_1*identity_y_1_D
D=C

eigenval_A,eigenvec_A=np.linalg.eig(A)
eigenval_C,eigenvec_C=np.linalg.eig(C)

D_A= np.diag(eigenval_A)
D_C= np.diag(eigenval_C)
P_inv_A = np.linalg.inv(eigenvec_A)
P_inv_C = np.linalg.inv(eigenvec_C)

matrix_1_eig=sp.sparse.kron(sp.sparse.diags(np.diag(D_A)),identity_x_1_D)+sp.sparse.kron(identity_x_1_D,sp.sparse.diags(np.diag(D_A)))
matrix_2_eig=sp.sparse.kron(sp.sparse.diags(np.diag(D_C)),identity_y_1_D)+sp.sparse.kron(identity_y_1_D,sp.sparse.diags(np.diag(D_C)))
prec_eig_arr=sp.sparse.kron(matrix_1_eig, identity_y_2_D)+sp.sparse.kron(identity_x_2_D,matrix_2_eig)
del matrix_1_eig
del matrix_2_eig

diag_prec_eig=sp.sparse.diags(1/prec_eig_arr.diagonal())
del prec_eig_arr







schur_A=schur(A)
schur_B=schur_A
schur_C=schur(C)
schur_D=schur_C

print("test 1")

matrix_1=sp.sparse.kron(sp.sparse.diags(schur_A[0].diagonal()),identity_x_1_D)+sp.sparse.kron(identity_x_1_D,sp.sparse.diags(schur_A[0].diagonal()))
matrix_2=sp.sparse.kron(sp.sparse.diags(schur_C[0].diagonal()),identity_y_1_D)+sp.sparse.kron(identity_y_1_D,sp.sparse.diags(schur_C[0].diagonal()))
prec_arr=sp.sparse.kron(matrix_1, identity_y_2_D)+sp.sparse.kron(identity_x_2_D,matrix_2)
# print("debug prec:",prec_arr)
del matrix_1
del matrix_2
diag_prec=sp.sparse.diags(1/prec_arr.diagonal())
del prec_arr

len_vector=N_x*N_x*N_y*N_y
print("test 2")

# mat_2=(sparse.kron(S_1,identity_y_2_D)+sparse.kron(identity_x_2_D,S_2))


As=[schur_A[1].T,schur_A[1].T,schur_C[1].T,schur_C[1].T]
Bs=[schur_A[1],schur_A[1],schur_C[1],schur_C[1]]
print("DEBUG SHAPE MATRIX:",eigenvec_C.shape, schur_C[1].T.shape,type(schur_C[1].T),type(eigenvec_C))

eigenvec_A=np.array(eigenvec_A)
eigenvec_C=np.array(eigenvec_C)
P_inv_A=np.array(P_inv_A)
P_inv_C=np.array(P_inv_C)
A_eig_list=[eigenvec_A,eigenvec_A,eigenvec_C,eigenvec_C]
B_eig_list=[P_inv_A,P_inv_A,P_inv_C,P_inv_C]

dimensie_lijst=[N_x_1,N_x_2,N_y_1,N_y_2]

kron_vec_time=[]
sp_solve_time=[]

#Num of times preconditioner is applied
prec_1_num=0




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
    #triangular solve
    # y_2=spsolve_triangular(mat_2,y_1,lower=False)
    y_2=diag_prec@y_1

    sp_solve_time.append(time.time()-start_time)
    y_3=kron_vec_prod(Bs,y_2)
    y_3=np.reshape(y_3,(len_vector,1))
    # del x
    # print("prec applied")
    return y_3

def prec_eigen_decomposition(x,*args):
    '''Dezelfde preconditioner alleen zijn de kronecker vector producten versneld nu'''
    global prec_1_num
    prec_1_num+=1
    x=np.reshape(x,(len_vector,1))
    # print("debug:", x.shape)
    y_1=kron_vec_prod(B_eig_list,x)
    y_2=diag_prec_eig@y_1
    y_3=kron_vec_prod(A_eig_list,y_2)
    y_3=np.reshape(y_3,(len_vector,1))
    return y_3



Hamiltonian_arr_1=[alpha_x*D_2_realline_x,identity_x_1_D,identity_y_1_D,identity_y_1_D]
Hamiltonian_arr_2=[identity_x_1_D,alpha_x*D_2_realline_x,identity_y_1_D,identity_y_1_D]
Hamiltonian_arr_3=[identity_x_1_D,identity_x_1_D,alpha_y*D_2_realline_y,identity_y_1_D]
Hamiltonian_arr_4=[identity_x_1_D,identity_x_1_D,identity_y_1_D,alpha_y*D_2_realline_y]

mat_vec=0
def mv(v):
    # print("hamiltonian apply")
    v_1=kron_vec_prod(Hamiltonian_arr_1,v)
    v_2=kron_vec_prod(Hamiltonian_arr_2,v)
    v_12=v_1+v_2
    v_3=kron_vec_prod(Hamiltonian_arr_3,v)
    v_4=kron_vec_prod(Hamiltonian_arr_4,v)
    v_34=v_3+v_4
    result_vec=v_12+v_34
    result_vec=np.reshape(result_vec,(len_vector,1))
    # print(result_vec.shape)
    # print(V.shape)
    v=v.reshape((len_vector,1))
    result_vec+=V@v
    global mat_vec
    mat_vec+=1
    # del v
    # del v_1
    # del v_2
    # del v_3
    # del v_4
    # del v_12
    # del v_34
    return result_vec

# S_1=sparse.kron(schur_A[0],identity_x_1_D)+sparse.kron(identity_x_1_D,schur_A[0])
# S_2=sparse.kron(schur_C[0],identity_y_1_D)+sparse.kron(identity_y_1_D,schur_C[0])
# mat_2=(sparse.kron(S_1,identity_y_2_D)+sparse.kron(identity_x_2_D,S_2))

def is_sparse_diagonally_dominant(matrix):
    num_rows, num_cols = matrix.shape
    for i in range(num_rows):
        row = matrix.getrow(i)
        diagonal_element = row[0, i]
        other_elements = row[0, row.indices != i]

        if abs(diagonal_element) < abs(other_elements).sum():
            return False

    return True

def diagonal_dominance_ratio(matrix):
    num_rows, num_cols = matrix.shape
    dominance_ratios = []
    for i in range(num_rows):
        row = matrix.getrow(i)
        diagonal_element = row[0, i]
        other_elements = row[0, row.indices != i]

        if other_elements.size == 0:
            # Handle the case where there are no off-diagonal elements in the row
            dominance_ratios.append(0)
        else:
            dominance_ratios.append(abs(diagonal_element) / abs(other_elements).sum())

    return max(dominance_ratios)

# print("diagonaal dominant?", is_sparse_diagonally_dominant(mat_2), "diagonaal dominantie ratio:", diagonal_dominance_ratio(mat_2))
# print("diagonaaldominat",diagonal_dominance_ratio(sp.sparse.csr_matrix(alpha_x*D_2_realline_x/E_target)))


def prec_eigen_exact(x,*args):
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
    # y_2=spsolve_triangular(mat_2,y_1,lower=False,overwrite_b=True)
    # y_2=solve_triangular(mat_2.toarray(),y_1,lower=False,overwrite_b=True,check_finite=False)

    sp_solve_time.append(time.time()-start_time)
    # y_3=mat_1.T@y_2
    y_2=kron_vec_prod(Bs,y_2)
    y_2=np.reshape(y_2,(len_vector,1))
    return y_2



return_vec=1


Hamiltonian_TISE = LinearOperator((N_x**2*N_y**2,N_x**2* N_y**2), matvec=mv)

num_tests=1
time_arr=[]

print("test3")

for i in range(num_tests):
    start_time=time.time()
    # eigenval,eigenvec=jdqr.jdqr(Hamiltonian_TISE/E_target,num=1,target=-target,tol=1e-10,return_eigenvectors=True,subspace_dimensions=(20,50),maxit=2500)
    # eigenval,eigenvec=jdqr.jdqr(Hamiltonian_TISE/E_target,num=1,target=-target,tol=1e-10,return_eigenvectors=True,subspace_dimensions=(20,50),maxit=2500,prec=prec_eigen_1)
    # jdqr.jdqr(Hamiltonian_TISE/E_target,num=1,target=Target.SmallestRealPart,tol=1e-10,arithmetic="complex",return_eigenvectors=False,subspace_dimensions=(5,15),maxit=2500,prec=prec_eigen_1)
    jdqr.jdqr(Hamiltonian_TISE/E_target,num=1,target=Target.SmallestRealPart,tol=1e-10,arithmetic="complex",return_eigenvectors=False,subspace_dimensions=(5,15),maxit=2500,prec=prec_eigen_decomposition)

# jdqr.jdqr(Hamiltonian_TISE/E_target,num=1,target=Target.SmallestRealPart,tol=1e-10,arithmetic="complex",return_eigenvectors=False,subspace_dimensions=(5,15),maxit=2500,prec=prec_eigen_exact)
    # jdqr.jdqr(Hamiltonian_TISE/E_target,num=1,target=Target.SmallestRealPart,tol=1e-10,arithmetic="complex",return_eigenvectors=False,subspace_dimensions=(20,30),maxit=2500)

    # jdqr.jdqr(Hamiltonian_TISE/E_target,num=1,target=Target.SmallestRealPart,arithmetic="complex",tol=1e-10,return_eigenvectors=False,subspace_dimensions=(20,50),maxit=2500)


    end_time=time.time()
    time_arr.append(end_time-start_time)

print("elapsed time:", np.mean(time_arr))

# print("kron vec prod time:", np.mean(kron_vec_time))
# print("spsolve time:", np.mean(sp_solve_time))


print("Preconditioner was called:", prec_1_num,"times")
print("Hamiltonian Operator was called", mat_vec,"times")

#N_x=20,N_y=10

#no prec
#mean time is 9 s
#hamiltonian was called 3905 times
#number iterations is 28

#prec
#mean time 5.23 seconds
#Hamiltonian was called 112 times
#preconditioner was called 110 times
#number of iterations is 25

#N_x=40,N_y=20
#prec: 83 seconden, 26 iteraties, hamiltonian 122 times, preconditioner 120 times
#no prec: 334 sec, 29 iteraties, hamiltonian called 889 times

#N_x=60, N_y=30
#prec: 26 iteraties, preconditioner 123 times, hamiltonian 125 times, 466 seconds
#No prec: 29 iteraties, hamiltonian 1497 times, 2053.0506930351257 seconds

# from sklearn.decomposition import TruncatedSVD
#
# svd = TruncatedSVD(n_components=5)
# W_reshaped=np.reshape(eigenvec[:,0],(N_x**2,N_y**2))
# # Transform your matrix using the fitted SVD
# matrix_reduced = svd.fit_transform(W_reshaped)
# matrix_restored = svd.inverse_transform(matrix_reduced)
#
# print("svd error:", np.linalg.norm(W_reshaped-matrix_restored))