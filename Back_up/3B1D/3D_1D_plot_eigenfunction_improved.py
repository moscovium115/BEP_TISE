
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
N_x=128
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
result_array_3= 1*Test_x[:, np.newaxis] +0* Test_y[np.newaxis, :]

# Print the result
rel_distance_1=np.abs(np.ravel(result_array_1))
rel_distance_2=np.abs(np.ravel(result_array_2))
rel_distance_3=np.abs(np.ravel(result_array_3))

#fermions dus ze moeten distinguishable zijn, dus er is potentiaal tussen hen
V_test=-v0*(np.exp(-(rel_distance_1**2))+np.exp(-(rel_distance_2**2))-0*np.exp(-(rel_distance_3**2)))
V=sp.sparse.diags(V_test)
A_1=alpha_x*D_2_realline_x
B_1=alpha_y*D_2_realline_y
from scipy.sparse.linalg import LinearOperator
#kronecker product optimaliseren
def mv(v):
    W=np.reshape(v,(N_x,N_y)).T
    test_vec_2=B_1@W+W@A_1.T
    test_vec=v.reshape((N_x*N_y,1))
    test_vec_2=np.reshape(test_vec_2.T,(N_x*N_y,1))+V@test_vec
    return test_vec_2

Hamiltonian_TISE= LinearOperator((N_x*N_y,N_x*N_y), matvec=mv)
target=2.71

sigma_1=-abs(alpha_y)*target/(abs(alpha_x+alpha_y))
sigma_2=-abs(alpha_x)*target/(abs(alpha_x+alpha_y))

#solving sylvester equation
A=(alpha_y*D_2_realline_y)/E_target-sigma_1*Id_y
B=(alpha_x*D_2_realline_x)/E_target-sigma_2*Id_x
B=B.T


schur_A=schur(A)
schur_B=schur(B)


def prec_func(x, *args):
    #function approximates w in (A-sigma*I)w=x
    C=np.reshape(x,(N_x,N_y))
    X = solve_sylvester(A,B, C.T,schur_A,schur_B)
    result_test=np.reshape(X.T,(N_x*N_y,1))
    return result_test

num_tests=1

test_arr=[]
for test in range(num_tests):
    start_time = time.time()
    eigenval,eigenvec=jdqr.jdqr(Hamiltonian_TISE/E_target,num=3,target=-target,arithmetic="complex",tol=1e-10,return_eigenvectors=True,subspace_dimensions=(20,50),maxit=2500,prec=prec_func)
    # eigenval,eigenvec=jdqr.jdqr(Hamiltonian_TISE/E_target,num=1,target=-2.71,tol=1e-10,return_eigenvectors=True,subspace_dimensions=(20,50),maxit=2500)
    end_time=time.time()
    test_arr.append(end_time-start_time)

print(np.mean(test_arr))
weights=eigenvec[:,0]


#Dit is sowieso niet de handigste manier om dit te programmeren.

W_reshaped_svd=np.reshape(np.abs(weights),(N_x,N_y),order='C')
x=np.repeat(mapping_x, N_y)
y=np.tile(mapping_y, N_x)
z=weights
#zodat we alleen de punten in de plot zien die we willen zien
x_max=30
y_max=30
# mask = (x < 0) | (x > x_max) | (y< 0) | (y > y_max)
# z[mask] = np.nan
print(weights[N_y:2*N_y])
# W_reshaped=np.reshape(np.abs(weights),(N_x,N_y),order='C')
W_reshaped=np.reshape(np.abs(weights),(N_x,N_y),order='C')

print("W",W_reshaped)
print(W_reshaped.shape)

X=np.reshape(x,(N_x,N_y),order='C')
Y=np.reshape(y,(N_x,N_y),order='C')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set(xlim3d=(0,x_max), xlabel='X')
ax.set(ylim3d=(0, y_max), ylabel='Y')
ax.view_init(elev=26, azim=46)

# Y, X = np.meshgrid(range(1, N_y+1), range(1, N_x+1))
print("debug X:", X)
print("debug Y:", Y)
surf = ax.plot_surface(X, Y, W_reshaped, cmap='viridis')
# plt.savefig("wave_eq.eps")


# Set the title
# ax.set_title(f'E={Lambda[i-1, i-1]:.4e}')

# Show the plot
plt.savefig("wave_eq_not_truncated.eps")
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
print((mapping_x))
print((mapping_y))

print(np.argmax(np.abs(weights)))
print(mapping_x[0],mapping_y[0])
from sklearn.decomposition import TruncatedSVD
num_sing_val=1
svd = TruncatedSVD(n_components=num_sing_val)

# Transform your matrix using the fitted SVD
matrix_reduced = svd.fit_transform(W_reshaped_svd)
matrix_restored = svd.inverse_transform(matrix_reduced)
print("shape", matrix_reduced.shape)

#svd
# from scipy.linalg import svd
# U, s, VT = svd(W_reshaped)
#
# print("singular values:", s)
#
# plt.plot(np.arange(len(s)),s,"o", markersize=10)
# plt.xlim([-1,10])
# plt.xlabel("$i$")
# plt.ylabel("$\sigma$")
# plt.savefig("singular_values.png")
# plt.savefig("singular_values.eps")
# plt.show()

#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # ax.title("truncated svd plot")
# ax.set(xlim3d=(0,x_max), xlabel='X')
# ax.set(ylim3d=(0, y_max), ylabel='Y')
# ax.view_init(elev=26, azim=46)
# X_masked = np.where((X <= x_max) & (Y <= y_max), X, np.nan)
# Y_masked = np.where((X <= x_max) & (Y <= y_max), Y, np.nan)
# W_svd_masked = np.where((X <= x_max) & (Y <= y_max), matrix_restored, np.nan)# Y, X = np.meshgrid(range(1, N_y+1), range(1, N_x+1))
# print("debug X:", X)
# print("debug Y:", Y)
# # matrix_restored[matrix_restored==420]=np.nan
# surf_1 = ax.plot_surface(X_masked, Y_masked, W_svd_masked, cmap='viridis')
# plt.savefig("wave_eq_truncated_svd"+"_"+str(num_sing_val)+".eps")
# plt.show()

#norm van de oplossingsvector is 1, door schrodinger vergelijking.
print("norm diference:",np.linalg.norm(matrix_restored-W_reshaped,ord="fro"))
print("norm original:",np.linalg.norm(W_reshaped,ord="fro"))


# U, s, VT = np.linalg.svd(W_reshaped.T)
# print(s)
# hmmm=D_2_realline_y@(VT.T)[:,0]/(VT.T)[:,0]
i=0
# hmmm=D_2_realline_y@(U[:,i])/(U[:,i])

# print(hmmm)
# print("gem:",np.mean(hmmm),"std:",np.std(hmmm))
# print((D_2_realline_y@U[:,0]-(np.mean(hmmm)*U[:,0]))/U[:,0])
# print(U[:,0])

# print(U[0,:])
# print(D_2_realline_x@U[:,10]-0.133680475*U[:,0])
# print(U[:,0])
#
# eigenval_1,eigenvectors=np.linalg.eig(alpha_x*D_2_realline_x/E_target)
# eigenval_1=np.sort(eigenval_1)
# # print(eigenval)
# eigenval_2,eigenvectors=np.linalg.eig(alpha_y*D_2_realline_y/E_target)
# # print(eigenval)
# eigenval_2=np.sort(eigenval_2)
#
# for num_1 in eigenval_1:
#     for num_2 in eigenval_2:
#         print(num_1+num_2,"test")

#
# print((Hamiltonian_TISE@weights)[0]/weights[0])
