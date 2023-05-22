import numpy as np
import numpy.ctypeslib as ctl
import ctypes

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt





libname="N_body.so"
libdir = './'
mylib=ctl.load_library(libname, libdir)

#number of chebyshev polynomials
N=100
cheb_nodes=np.cos(np.pi*(2*np.arange(1,N+1)-1)/(2*N))


###Initialize the first derivative matrix
Initialize_First_Derivative_function=mylib.Init_First_derivative_matrix
Initialize_First_Derivative_function.restype = ctypes.c_longlong
Initialize_First_Derivative_function.argtypes = [ctl.ndpointer(np.float64,flags='aligned, c_contiguous'),ctl.ndpointer(np.float64,flags='aligned, c_contiguous'),ctypes.c_int]
Dirac_1=np.ones((N,N))
#changes Dirac_1 in place
Initialize_First_Derivative_function(Dirac_1,cheb_nodes,N)


###Initialize the second derivative matrix
Initialize_Second_Derivative_function=mylib.Init_Second_derivative_matrix
Initialize_Second_Derivative_function.restype = ctypes.c_longlong
Initialize_Second_Derivative_function.argtypes = [ctl.ndpointer(np.float64,flags='aligned, c_contiguous'),ctl.ndpointer(np.float64,flags='aligned, c_contiguous'),ctypes.c_int]

Dirac_2=np.ones((N,N))
Initialize_Second_Derivative_function(Dirac_2,cheb_nodes,N)
print(Dirac_2)



L=1

A=np.diag((1/L)*(1-cheb_nodes**2)**(3/2))
#B matrix
B=np.diag((-3/L**2)*cheb_nodes*(1-cheb_nodes**2)**(2))

mapping=L*cheb_nodes/np.sqrt((1-cheb_nodes**2))

alpha=-0.5
# h_bar=1.0545718e-34
# m=9.10938356e-31
# alpha=-h_bar**2/(2*m)
#onderste potentiaal is voor de harmonische oscillator
# pot_arr=0.5*mapping**2
# inf=1000000000000000000
# pot_arr=inf*np.ones(N)
# pot_arr[mapping>=0]=0
# pot_arr[mapping>=1]=inf
# omega=0.01
# pot_arr=0.5*mapping**2
# pot_arr=-0.34459535*np.exp(-mapping**2)
pot_arr=-0.08887372*np.exp(-mapping**2)

print("potentieel:",pot_arr)
V=np.diag(pot_arr)

D_2_realline=A@A@Dirac_2+B@Dirac_1

Hamiltonian_TISE=alpha*D_2_realline+V




###CHEB FUNCTIONS

def chebyshev_polynomial(n,x):
    if n==0:
        return 1
    elif n==1:
        return x
    else:
        return 2*x*chebyshev_polynomial(n-1,x)-chebyshev_polynomial(n-2,x)

##Recursively constructing Chebyshev polynomial derivatives
def chebyshev_polynomial_derivative(n,x):
    if n==0:
        return 0
    elif n==1:
        return 1
    else:
        return 2*(chebyshev_polynomial(n-1,x)+x*chebyshev_polynomial_derivative(n-1,x))-chebyshev_polynomial_derivative(n-2,x)

##Recursively constructing Chebyshev polynomial second derivatives
def chebyshev_polynomial_second_derivative(n,x):
    if n==0:
        return 0
    elif n==1:
        return 0
    else:
        return 2*(2*chebyshev_polynomial_derivative(n-1,x)+x*chebyshev_polynomial_second_derivative(n-1,x))-chebyshev_polynomial_second_derivative(n-2,x)

##Constructing rational Chebyshev polynomials, which are functions C_j(x)=T_n(x)/(T_n'(x_j)*(x-x_j)) , where T_n(x) is the Chebyshev polynomial of degree n
def chebyshev_rational(n,x,j):
    arr_1=np.zeros(len(x))

    for i in range(len(arr_1)):
        if (x[i]-cheb_nodes[j])==0:
            arr_1[i]=1
        else:
            arr_1[i]=chebyshev_polynomial(n,x[i])/((x[i]-cheb_nodes[j])*chebyshev_polynomial_derivative(n,cheb_nodes[j]))
    return arr_1


#Linear combination of rational Chebyshev polynomials, weights are found by solving the eigenvalue problem
def chebyshev_approximation_rational(weights,x):
    func=0
    for i in range(len(weights)):
        func+=chebyshev_rational(N,x,i)*weights[i]
    return func


###


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
# def plot_eigenfunction(weights_TISE):
#     #plotting the eigenfunction
#     domain=np.linspace(start_x,end_x,300)
#
#     Cheb_approx_cpp=mylib.Approximation_Chebyshev
#     Cheb_approx_cpp.restype = ctypes.c_longlong
#     Cheb_approx_cpp.argtypes = [ctl.ndpointer(np.float64,flags='aligned, c_contiguous'), ctl.ndpointer(np.float64,flags='aligned, c_contiguous'),
#                                 ctl.ndpointer(np.float64,flags='aligned, c_contiguous'), ctl.ndpointer(np.float64,flags='aligned, c_contiguous'),ctypes.c_int,ctypes.c_int]
#
#     func=np.zeros(len(domain))
#     weights_TISE=np.array(weights_TISE)
#     print(Cheb_approx_cpp(weights_TISE,func,cheb_nodes,domain,len(domain),N))
#     print("func resultaat", func)
#
#     amplitude_gaussinan=np.sqrt(m*omega/(np.pi*h_bar))
#     dx=(end_x-start_x)/len(domain)
#     sq_func=np.square(func)
#     plt.plot(domain,sq_func/((dx*np.sum(sq_func))))
#     plt.plot(domain,amplitude_gaussinan*np.square(np.exp(-m*omega*np.square(domain)/(2*h_bar))))
#     plt.show()

# plot_eigenfunction(weights_TISE)
#
