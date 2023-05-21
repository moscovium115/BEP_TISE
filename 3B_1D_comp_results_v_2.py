#this script is made to reproduce the 1D results from the "tensor products" paper
import numpy as np
import numpy.ctypeslib as ctl
import ctypes
from scipy import sparse
import scipy as sp
import matplotlib
matplotlib.use('TkAgg')



M,m=20,1
mass_ratio=M/m
alpha_x=-1/((1+mass_ratio))
alpha_y=(1+2*mass_ratio)/(-4*(1+1*mass_ratio))
energy_level=2

if energy_level==1:
    E_target=1e-1
    v0=0.34459535
elif energy_level==2:
    E_target=1e-2
    v0=0.08887372
else:
    E_target=1e-3
    v0=0.02613437




import matplotlib.pyplot as plt
libname="N_body.so"
libdir = './'
mylib=ctl.load_library(libname, libdir)


N_y=150
N_x=int(N_y/2)
cheb_nodes_x=np.cos(np.pi*(2*np.arange(1,N_x+1)-1)/(4*N_x))
print(cheb_nodes_x)
cheb_nodes_y=np.cos(np.pi*(2*np.arange(1,N_y+1)-1)/(4*N_y))
cheb_nodes_x_original=np.cos(np.pi*(2*np.arange(1,2*N_x+1)-1)/(4*N_x))
cheb_nodes_y_original=np.cos(np.pi*(2*np.arange(1,2*N_y+1)-1)/(4*N_y))
print(cheb_nodes_x_original)
print(2*np.arange(1,2*8+1)-1)



###Initialize the first derivative matrix
Initialize_First_Derivative_function=mylib.Init_First_derivative_matrix
Initialize_First_Derivative_function.restype = ctypes.c_longlong
Initialize_First_Derivative_function.argtypes = [ctl.ndpointer(np.float64,flags='aligned, c_contiguous'),ctl.ndpointer(np.float64,flags='aligned, c_contiguous'),ctypes.c_int]
#checkpoint
Dirac_1_x=np.ones((2*N_x,2*N_x))
Dirac_1_y=np.ones((2*N_y,2*N_y))
#changes Dirac_1 in place

#dirac 1 klopt nu voor gaussisch
Initialize_First_Derivative_function(Dirac_1_x,cheb_nodes_x_original,2*N_x)
Initialize_First_Derivative_function(Dirac_1_y,cheb_nodes_y_original,2*N_y)

print("Dirac 1 x:", Dirac_1_x[0,:])
print(Dirac_1_x.shape)

###Initialize the second derivative matrix
Initialize_Second_Derivative_function=mylib.Init_Second_derivative_matrix
Initialize_Second_Derivative_function.restype = ctypes.c_longlong
Initialize_Second_Derivative_function.argtypes = [ctl.ndpointer(np.float64,flags='aligned, c_contiguous'),ctl.ndpointer(np.float64,flags='aligned, c_contiguous'),ctypes.c_int]

Dirac_2_x=np.ones((2*N_x,2*N_x))
Dirac_2_y=np.ones((2*N_y,2*N_y))
Initialize_Second_Derivative_function(Dirac_2_x,cheb_nodes_x_original,2*N_x)
Initialize_Second_Derivative_function(Dirac_2_y,cheb_nodes_y_original,2*N_y)
print("dirac 2:", Dirac_2_y)
print("shape:", Dirac_2_y.shape)




L_y=3*(1/np.sqrt(2*E_target))*(2+1/mass_ratio)/np.sqrt(1+1/mass_ratio)
L_x=1.5*(1/np.sqrt(2*E_target))*(2+1/mass_ratio)/np.sqrt(1+1/mass_ratio)


A_x=np.diag((1/L_x)*(1-cheb_nodes_x_original**2)**(3/2))
A_y=np.diag((1/L_y)*(1-cheb_nodes_y_original**2)**(3/2))
#B matrix
B_x=np.diag((-3/L_x**2)*cheb_nodes_x_original*(1-cheb_nodes_x_original**2)**(2))
B_y=np.diag((-3/L_y**2)*cheb_nodes_y_original*(1-cheb_nodes_y_original**2)**(2))

mapping_x=L_x*cheb_nodes_x/np.sqrt((1-cheb_nodes_x**2))
print("x mapping:", mapping_x)
mapping_y=L_y*cheb_nodes_y/np.sqrt((1-cheb_nodes_y**2))


D_2_realline_x=A_x@A_x@Dirac_2_x+B_x@Dirac_1_x
print("debug D2 matrix:",D_2_realline_x[0:N_x,np.arange(2*N_x-1,N_x-1,step=-1)])
#we maken de D2 matrix kleiner, en bij de tweede term die gesommeerd wordt, vervangen we de kolommen van volgorde
D_2_realline_x=D_2_realline_x[0:N_x,0:N_x]+D_2_realline_x[0:N_x,np.arange(2*N_x-1,N_x-1,step=-1)]

D_2_realline_y=A_y@A_y@Dirac_2_y+B_y@Dirac_1_y
D_2_realline_y=D_2_realline_y[0:N_y,0:N_y]+D_2_realline_y[0:N_y,np.arange(2*N_y-1,N_y-1,step=-1)]

print("D_2 real line x: ",D_2_realline_x)


Id_y=np.identity(N_y)
Id_x=np.identity(N_x)


Test_x=mapping_x
Test_y=mapping_y

# Convert the result list to a NumPy array
# result_array = np.array(result)
# Perform broadcasting to compute the result
result_array_1 = Test_x[:, np.newaxis] + 0.5 * Test_y[np.newaxis, :]
result_array_2 = Test_x[:, np.newaxis] - 0.5 * Test_y[np.newaxis, :]

# Print the result
print(np.ravel(result_array_1))
rel_distance_1=np.abs(np.ravel(result_array_1))
rel_distance_2=np.abs(np.ravel(result_array_2))

test_potential=np.zeros(N_x*N_y)

for i in range(N_x):
    for j in range(N_y):
        val_b=(0.5*mapping_x[i]+mapping_y[j])**2
        val_c=(0.5*mapping_x[i]-mapping_y[j])**2
        test_potential[(i)*N_y+j]=-v0*(np.exp(-val_b)+np.exp(-val_c))
        print((i)*N_x+j)

print("test potential:",test_potential)


print("rel distance arr:", rel_distance_1)

V=-v0*(np.exp(-(rel_distance_1**2))+np.exp(-(rel_distance_2**2)))
V=sp.sparse.diags(test_potential)

print("V:",V)
print("V:",V.shape)



#kronecker product optimaliseren
Hamiltonian_TISE=alpha_x*sparse.kron(D_2_realline_x,Id_y)+(alpha_y)*sparse.kron(Id_x, D_2_realline_y)+V

print("hamiltonian shape:", Hamiltonian_TISE.shape)



from jadapy import jdqr
from jadapy import Target


from scipy.linalg import solve_sylvester
from scipy.linalg import schur, eigvals

sigma_1=-abs(alpha_y)*2.71/(abs(alpha_x+alpha_y))
sigma_2=-abs(alpha_x)*2.71/(abs(alpha_x+alpha_y))


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

import time


num_tests=10

test_arr=[]
#checkpoint
for test in range(num_tests):
    start_time = time.time()
    # eigenval,eigenvec=jdqr.jdqr(Hamiltonian_TISE/E_target,num=1,target=-2.71,tol=1e-10,return_eigenvectors=True,subspace_dimensions=(20,50),maxit=2500,prec=prec_func)

    eigenval,eigenvec=jdqr.jdqr(Hamiltonian_TISE/E_target,num=1,target=-2.71,tol=1e-10,return_eigenvectors=True,subspace_dimensions=(20,50),maxit=2500)
    end_time=time.time()
    test_arr.append(end_time-start_time)


print("running time:",np.mean(test_arr))

#Elke keer 10 tests

#bij N_y=80, N_x=40, epsilon_2=0.10
#prec 0.4247476577758789
#no prec 1.4719385147094726

#bij N_y=100, N_x=50, epsilon_2=0.10
#prec 0.760047
#no prec 1.6864

#Bij N_y=150
#prec 3.4252004384994508
#no prec 62.98380177021026



print("eigenval:",np.real(eigenval[0]))
print("all eigenval",np.real(eigenval))
print(len(eigenvec))
print(eigenvec[N_y-1])
print(eigenvec[0])
print(mapping_y[0],mapping_y[N_y-1])
num_eigenvec=0
print(eigenvec.shape)

print(eigenvec[1,num_eigenvec],eigenvec[N_y+2,num_eigenvec])

start_vec=np.ones((N_x*N_y,1))
start_time = time.time()
vec_1=Hamiltonian_TISE*start_vec
end_time=time.time()
time_1=end_time-start_time
print("running time matrix vector product:",time_1)

start_time = time.time()
W=np.reshape(start_vec,(N_y,N_x))
product_1=alpha_y*(D_2_realline_y @ W)+alpha_x*(W @ D_2_realline_x.T)
print("product_1:",product_1.shape)
vec_2=np.reshape(product_1,(N_x*N_y,1))+V@start_vec
end_time=time.time()
time_2=end_time-start_time
print("running time operator form",time_2)

#deze snelheidsverbeteringen tellen erg op bij het berekenen van de eigenwaarden
print(100*(time_1-time_2)/time_2, "% sneller")

#ik merk kleine numerieke verschillen op tussen de twee methodes, maar die zijn onder machine precision dus wss hoef k me er geen zorgen over te maken? check met jonas