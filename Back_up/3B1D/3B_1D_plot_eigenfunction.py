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
import matplotlib.pyplot as plt


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


N_x=100
N_y=int(N_x/2)

#deze code onderin klopt nu
#de verwarring ontstond vgm omdat voor het potentiaal het teken van x en y niet uitmaakt, alleen de fase die ze hebben t.o.v. elkaar
#dus als je wilt plotten gebruik cheb_nodes_x_original[0::2] dan neem je de negatieve punten ook mee,
#hierdoor is de forloop voor potentiaal berkenen vgm stuk sneller.
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
B=B.T
A=np.asarray(A, dtype=np.float64)
B=np.asarray(B, dtype=np.float64)

schur_A=schur(A)
schur_B=schur(B)


def prec_func(x, *args):
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
    # eigenval,eigenvec=jdqr.jdqr(Hamiltonian_TISE/E_target,num=1,target=-2.71,tol=1e-10,return_eigenvectors=True,subspace_dimensions=(20,50),maxit=2500)
    end_time=time.time()
    test_arr.append(end_time-start_time)

print(np.mean(test_arr))
weights=eigenvec[:,0]


#Dit is sowieso niet de handigste manier om dit te programmeren.


x=np.repeat(np.arange(N_x), N_y)
y=np.tile(np.arange(N_y), N_x)
z=weights
print(weights[N_y:2*N_y])
W_reshaped=np.reshape(weights,(N_x,N_y),order='C')
print("W",W_reshaped)
print(W_reshaped.shape)

X=np.reshape(x,(N_x,N_y),order='C')
Y=np.reshape(y,(N_x,N_y),order='C')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Y, X = np.meshgrid(range(1, N_y+1), range(1, N_x+1))
print("debug X:", X)
print("debug Y:", Y)
surf = ax.plot_surface(Y, X, W_reshaped, cmap='viridis')

# Set the title
# ax.set_title(f'E={Lambda[i-1, i-1]:.4e}')

# Show the plot
plt.show()




# a=1
# # Create a scatter plot with interpolated shading
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# sc = ax.scatter(x, y, z, c=z, cmap='viridis', marker='o', s=5)
# # plt.colorbar(sc, label='z-values')
#
# # Set labels and title
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set(xlim3d=(0,a*N_x), xlabel='X')
# ax.set(ylim3d=(0, a*N_y), ylabel='Y')
# ax.set(zlim3d=(a*np.min(z), a*np.max(z)), zlabel='Z')
# ax.set_title('Interactive 3D Scatter Plot')
#
# plt.show()
# # print(weights)














# sample=-1
# view_earth=0
# ax.set(xlim3d=(-a*L_x,a*L_x), xlabel='X')
# ax.set(ylim3d=(-a*L_y, a*L_y), ylabel='Y')
# ax.set(zlim3d=(a*np.min(z), a*np.max(z)), zlabel='Z')
# ax.view_init(elev=20, azim=-90)
# ax.legend()
# ax.plot(x,y,z,'o',label="eigenfunction",markersize=0.5, shading='interp')
# plt.legend()
# plt.show()

# Creating the Animation object
# ani = animation.FuncAnimation(fig, update_lines, num_steps, fargs=(walks, lines), interval=1, repeat=False)


print(max(np.abs(weights)))
print(max(mapping_x))
print(max(mapping_y))
