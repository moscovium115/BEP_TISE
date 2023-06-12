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
N_x=10
N_y=int(N_x/2)

# N_x=6
# N_y=N_x


# print("storage hamiltonian matrix:", 8*(N_x**4 * N_y**4 )/1e9, "GB")

N_x_1=N_x
N_x_2=N_x
N_y_1=N_y
N_y_2=N_y

cheb_nodes_x=np.cos(np.pi*(2*np.arange(1,N_x+1)-1)/(2*N_x))
cheb_nodes_y=np.cos(np.pi*(2*np.arange(1,N_y+1)-1)/(2*N_y))
# print("cheb nodes x:", cheb_nodes_x)
cheb_nodes_x_original=np.cos(np.pi*(2*np.arange(1,2*N_x_1+1)-1)/(4*N_x_1))
cheb_nodes_y_original=np.cos(np.pi*(2*np.arange(1,2*N_x_1+1)-1)/(4*N_x_1))

#constructing the derivative matrices
Dirac_1_x=construct_first_derivative_matrix(cheb_nodes_x,N_x_1)
Dirac_2_x=construct_second_derivative_matrix(cheb_nodes_x,N_x_1)


Dirac_1_y=construct_first_derivative_matrix(cheb_nodes_y,N_y_1)
Dirac_2_y=construct_second_derivative_matrix(cheb_nodes_y,N_y_1)



L_x=1*0.5*(1/np.sqrt(2*E_target))*(2+1/mass_ratio)/np.sqrt(1+1/mass_ratio)
L_y=1*0.25*(1/np.sqrt(2*E_target))*(2+1/mass_ratio)/np.sqrt(1+1/mass_ratio)
# print("L_x:", L_x)
# print("L_y:", L_y)
A_x=np.diag((1/L_x)*(1-cheb_nodes_x**2)**(3/2))
B_x=np.diag((-3/L_x**2)*cheb_nodes_x*(1-cheb_nodes_x**2)**(2))

A_y=np.diag((1/L_y)*(1-cheb_nodes_y**2)**(3/2))
B_y=np.diag((-3/L_y**2)*cheb_nodes_y*(1-cheb_nodes_y**2)**(2))

mapping_x=L_x*cheb_nodes_x/np.sqrt((1-cheb_nodes_x**2))
mapping_y=L_y*cheb_nodes_y/np.sqrt((1-cheb_nodes_y**2))

#vermijd A_x @A_x en stel meteen de A matrix op met de kwadraten
D_2_realline_x=A_x@A_x@Dirac_2_x+B_x@Dirac_1_x
#Waarom wordt de selection rule niet meer gebruikt in het 2D geval

# D_2_realline_x=D_2_realline_x[0:N_x,0:N_x]+D_2_realline_x[0:N_x,np.arange(2*N_x-1,N_x-1,step=-1)]

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

#setting up the potential matrix
#it is equal to the matlab implementation

#TODO: For loop vectoriseren

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


# print("potential matrix:", V_arr)
V=sp.sparse.diags(V_arr)

Hamiltonian_TISE+=V



from jadapy import jdqr
from jadapy import Target





def prec_func(x, *args):
    #function approximates w in (A-sigma*I)w=x
    res_1=M.dot(x)


    return res_1


target=2.71
sigma_1=-abs(alpha_y)*target/(abs(alpha_x+alpha_y))
sigma_2=-abs(alpha_x)*target/(abs(alpha_x+alpha_y))


#solving sylvester equation
A_1=alpha_y*D_2_realline_y/E_target-sigma_1*np.identity(N_y)
B_1=alpha_x*D_2_realline_x/E_target-sigma_2*np.identity(N_x)
B_1=B_1.T
A_1=np.asarray(A_1, dtype=np.float128)
B_1=np.asarray(B_1, dtype=np.float128)

schur_A_1=schur(A_1)
schur_B_1=schur(B_1)

#los sylvester equation op met tensor structuu  r

def sylvester_1D_y(x,first_matrix,second_matrix,schur_1,schur_2):
    #function approximates w in (A-sigma*I)w=x
    X_i=np.reshape(x,(N_x_1,N_x_2))
    G_i = solve_sylvester(first_matrix,second_matrix.T, X_i.T,schur_1,schur_2)
    result_test=np.reshape(G_i.T,(-1,1))
    # print("residual:",first_matrix@(G_i.T)+G_i.T@(second_matrix.T))
    return result_test

A=alpha_x*D_2_realline_x/E_target-sigma_2*identity_x_1_D
B=A
C=alpha_y*D_2_realline_y/E_target-sigma_1*identity_y_1_D
D=C
E=sparse.kron(C,identity_y_1_D)+sparse.kron(identity_y_1_D,D)
F=sparse.kron(A,identity_x_1_D)+sparse.kron(identity_x_1_D,B)


#deze weghalen later omdat te veel geheugen kost
E=E.toarray()
F=F.toarray()


schur_A=schur(A)
schur_B=schur(B)
print("U:",schur_A[0].diagonal())

schur_C=schur(C)
schur_D=schur(D)


def Sylv_1_d_prec_1(x,A,B,schur_A,schur_B):
   # C=np.reshape(x,(B.shape[0],A.shape[0]))
    #A=np.asarray(A, dtype=np.float128)
    #B=np.asarray(B, dtype=np.float128)
    #C=np.asarray(C, dtype=np.float128)
    # C=np.random.random((B.shape[0],A.shape[0]))
    # print(A)

    #X=solve_sylvester(A,B,C,schur_A,schur_B)
    #residu=A@X+X@B.T
    # print("mat_1:",residu)
    # print("mat_2:",C)
    # print("residual:",np.min(residu),np.max(residu))
   # A=alpha_x*D_2_realline_x/E_target-sigma_2*identity_x_1_D
   # B=A
   # schur_A=schur(A)
   # schur_B=schur(B)


   C=np.reshape(x,(A.shape[0],B.shape[1]))
   #C=np.asarray(C, dtype=np.float64)
   X = solve_sylvester(A,B, C,schur_A,schur_B)
   result_test=np.reshape(X.T,(A.shape[0]*B.shape[1],1))
   print("residual prec 1d:",np.linalg.norm(A@X+X@B.T-C))
   # print("residual 1:", A@X+X@B.T)
   # print("residual 2:", C)

   return np.ravel(result_test)


test_vec=np.random.random((N_x*N_x,1))
for i in range(100):
    test_sylv=Sylv_1_d_prec_1(np.ones((N_x*N_x,1)),A,B,schur_A,schur_B)
    print("test sylv:",i)







# def Sylv_1_d_prec_1(x,A,B,schur_A,schur_B):
#     """werkt alleen als de matrices A en B even groot zijn, maar dat maakt niet veel uit in dit geval"""
#     C_1=np.reshape(x,(A.shape[0],B.shape[0]))
#     C_1=np.asarray(C_1, dtype=np.float64)
#     X = solve_sylvester(A,B, C_1.T,schur_A,schur_B)
#     solution_vector=np.reshape(X.T,(A.shape[0]*B.shape[0],1))
#     # print("shape vec", solution_vector.shape)
#     print("res 1d prec:",np.linalg.norm(A@X+X@B.T-C_1.T))
#     return np.ravel(solution_vector)

#prec 2d functi is inefficient, omdat die nul matrices maakt die super groot gaan worden



def prec_2d(x,*args):
    """Solves (F (x) E ) y=x for y, where (x) is the Kronecker product."""
    Right_vector=x
    Right_vector_matrix=np.reshape(Right_vector,(E.shape[0],F.shape[0]),order='F')

    G=np.zeros((E.shape[0],F.shape[0]))

    for i in range(Right_vector_matrix.shape[1]):
        # g_i=np.linalg.solve(E,Right_vector_matrix[:,i])
        g_i=Sylv_1_d_prec_1(Right_vector_matrix[:,i],C,D,schur_C,schur_D)
        # print("res 2d preec:",np.linalg.norm( Right_vector_matrix[:,i]-E@g_i))
        G[:,i]=g_i

    Vectorisation_mat=np.zeros((F.shape[0],E.shape[0]))

    for i in range(Right_vector_matrix.shape[0]):
        # g_i_t=np.linalg.solve(F,G[i,:])
        g_i_t=Sylv_1_d_prec_1(G[i,:],A,B,schur_A,schur_B)
        Vectorisation_mat[:,i]=g_i_t

    #van matrix naar vector gaan, voert al een transpose uit, tenzij je "order=F" meegeeft
    vector=np.reshape(Vectorisation_mat.T,(-1,1),order='F')
    return vector

print("determinant F",np.linalg.det(F))
#A=alpha_x*D_2_realline_x
#B=A
#C=alpha_y*D_2_realline_y
#D=C
#E=sparse.kron(C,identity_y_1_D)+sparse.kron(identity_y_1_D,D)
#F=sparse.kron(A,identity_x_1_D)+sparse.kron(identity_x_1_D,B)


#dus de inverse moet zo: inverse=sp.sparse.linalg.inv(Hamiltonian_TISE/E_target+target*sp.sparse.eye(Hamiltonian_TISE.shape[0]))
#het moet geen -target zijn maar + target, nu kun je die 2d prec uitproberen

#onderstaande inverse werkt super
#matas=(Hamiltonian_TISE)/E_target+target*sp.sparse.eye(Hamiltonian_TISE.shape[0])
#inverse=sp.sparse.linalg.inv(matas)

#werkt slecht helaas, dit is de aanpak die Jonas had uitgelegd, maar die lijkt niet te werken
# inverse=sp.sparse.linalg.inv((sparse.kron(F,E))/E_target+target*sp.sparse.eye(Hamiltonian_TISE.shape[0]))



#def test_2d_prec(x,*args):
    # print("shapes debug opnieuw:",E.shape,F.shape)
 #   sol=inverse@x
  #  return sol
def woodbury_kronecker_3(E,F):
    E_inv=sparse.linalg.inv(E)
    mat_1=sparse.kron(E_inv,sparse.eye(F.shape[0]))
    result_matrix=sparse.kron(E_inv,F)
    inv_res=sparse.linalg.inv(result_matrix/1000)
    #benadering van inverse (result_matrix+ I) berekenen
    # mat_3=1000*inv_res-1000*inv_res@inv_res
    print("comng condition")
    print(result_matrix)
    #de echte inverse van (result_matrix+ I) berekenen
    print("norm mat_3")
    # mat_3=result_matrix+1*sparse.eye(E.shape[0]*F.shape[0])
    mat_3=result_matrix+1*sparse.eye(E.shape[0]*F.shape[0])

    # mat_3=mat_3.toarray()
    # mat_3=np.linalg.inv(mat_3)
    mat_3=sparse.linalg.inv(mat_3)
    print(np.linalg.norm(result_matrix.toarray()/100))
    result_matrix=sparse.eye(E.shape[0]*F.shape[0])-mat_3@result_matrix
    result_matrix=mat_1@result_matrix
    return result_matrix

def woodbury_kronecker_4(E,F,target):
    E_til=E+target/2*sparse.eye(E.shape[0])
    F_til=F+target/2*sparse.eye(F.shape[0])
    return woodbury_kronecker_3(E_til,F_til)


mat_1=alpha_x*sp.sparse.kron(D_2_realline_x,identity_x_1_D)+alpha_x*sp.sparse.kron(identity_x_1_D,D_2_realline_x)
mat_2=alpha_y*sp.sparse.kron(D_2_realline_y,identity_y_1_D)+alpha_y*sp.sparse.kron(identity_y_1_D,D_2_realline_y)
L_mat=sp.sparse.kron(mat_1,identity_y_2_D)+sp.sparse.kron(identity_x_2_D,mat_2)+V
L_mat=L_mat/E_target+target*sp.sparse.eye(L_mat.shape[0])
#L_mat=L_mat.toarray()



#woodbury matrix is hetzelfde als compar matrix, maar als ik potentiaal erin verwerkt klopt t niet meer snap niet wrm
#ik kan t daardoor niet gelijk krijgen aan de inverse van de H/E+tau I
# woodbury_matrix=woodbury_kronecker_4(mat_1/E_target,mat_2/E_target,target)
# woodbury_matrix=woodbury_matrix.toarray()


# compar_matrix=sparse.linalg.inv((sparse.kron(mat_1,np.eye(mat_2.shape[0]))+sparse.kron(np.eye(mat_1.shape[0]),mat_2)+V)/E_target+target*sparse.eye(mat_1.shape[0]*mat_2.shape[0]))
# compar_matrix=compar_matrix.toarray()
# print("res woodbury matrix:")
# print(np.linalg.norm(woodbury_matrix-compar_matrix))
# print(np.linalg.norm(inverse-woodbury_matrix))
# print(np.linalg.norm(inverse-compar_matrix))



#
# def test_2d_prec_2(x,*args):
#     x=np.reshape(x,(-1,1))
#
#     # print("woodbury matrix shape:",woodbury_matrix.shape)
#     # print("x shape:",x.shape)
#     woodbury_solution=woodbury_matrix@x
#     # print("woodbury solution shape:",woodbury_solution.shape)
#     # print("woodbury solution shape:",woodbury_solution.shape)
#     return woodbury_solution


num_tests=1
time_arr=[]

###################################
#hele goede preconditioner waarvoor we alleen maar 4keer de inverse van de 1d matrix hoeven te berekenen, dus A,B,C en D, en omdat A=B en, C=D, hoeven we maar 2 keer de inverse te berekenen
from scipy.optimize import minimize


mat_1_NKP=alpha_x*sp.sparse.kron(D_2_realline_x,identity_x_1_D)+alpha_x*sp.sparse.kron(identity_x_1_D,D_2_realline_x)
mat_2_NKP=alpha_y*sp.sparse.kron(D_2_realline_y,identity_y_1_D)+alpha_y*sp.sparse.kron(identity_y_1_D,D_2_realline_y)
L_mat_NKP=sp.sparse.kron(mat_1,identity_y_2_D)+sp.sparse.kron(identity_x_2_D,mat_2)

# L_mat_NKP=L_mat/E_target+target*sp.sparse.eye(L_mat.shape[0])
# L_mat_NKP=L_mat.toarray()
from scipy.optimize import basinhopping


def construct_approx(list):
    mat_1=list[0]*alpha_y*D_2_realline_y + list[1]*alpha_x*D_2_realline_x+list[2]*alpha_x*D_2_realline_x+list[3]*alpha_y*D_2_realline_y
    mat_2=list[4]*alpha_y*D_2_realline_y + list[5]*alpha_x*D_2_realline_x+list[6]*alpha_x*D_2_realline_x+list[7]*alpha_y*D_2_realline_y
    mat_3=list[8]*alpha_y*D_2_realline_y + list[9]*alpha_x*D_2_realline_x+list[10]*alpha_x*D_2_realline_x+list[11]*alpha_y*D_2_realline_y
    mat_4=list[12]*alpha_y*D_2_realline_y + list[13]*alpha_x*D_2_realline_x+list[14]*alpha_x*D_2_realline_x+list[15]*alpha_y*D_2_realline_y
    approximated_matrix=np.kron(mat_1,mat_2)
    approximated_matrix=np.kron(approximated_matrix,mat_3)
    approximated_matrix=np.kron(approximated_matrix,mat_4)
    return approximated_matrix
def objective(x):
    approximated_matrix=construct_approx(x)
    diff_matrix=L_mat_NKP-(approximated_matrix)
    return np.linalg.norm(diff_matrix,ord=1)

#order_H_matrix=4
# Initial guess
#x0 = np.ones(4*order_H_matrix)
# x0=np.random.random(2*order_H_matrix)


# Call the minimize function with Nelder-Mead method
# result = minimize(objective, x0,method="Nelder-Mead", tol=1e-10)
# result= basinhopping(objective, x0, niter=20)
# Print the optimized results
# print(result)

# print("approximated matrix:",construct_approx(result.x)-L_mat_NKP)

# approximated_matrix=construct_approx(result.x)
# print("diff approx hamiltonian:",np.linalg.norm(approximated_matrix-L_mat_NKP))
# approximated_matrix=np.linalg.inv(approximated_matrix/E_target+target*np.eye(approximated_matrix.shape[0]))

def construct_inverse_approx(list):
    mat_1=list[0]*alpha_y*D_2_realline_y + list[1]*identity_y_1_D
    mat_1=np.linalg.inv(mat_1)
    mat_2=list[2]*alpha_y*D_2_realline_y + list[3]*identity_y_1_D
    mat_2=np.linalg.inv(mat_2)
    mat_3=list[4]*alpha_x*D_2_realline_x + list[5]*identity_x_1_D
    mat_3=np.linalg.inv(mat_3)
    mat_4=list[6]*alpha_x*D_2_realline_x + list[7]*identity_x_1_D
    mat_4=np.linalg.inv(mat_4)
    approximated_matrix=np.kron(mat_1,mat_2)
    approximated_matrix=np.kron(approximated_matrix,mat_3)
    approximated_matrix=np.kron(approximated_matrix,mat_4)
    return approximated_matrix

# approximated_inverse_matrix=construct_inverse_approx(result.x)

# print("asa",np.linalg.norm(approximated_inverse_matrix))

# approximated_matrix=np.linalg.inv(approximated_matrix/E_target+target*np.eye(approximated_matrix.shape[0]))

# approximated_matrix=approximated_inverse_matrix



#inverted_MAT_1=sp.sparse.linalg.inv( mat_1_NKP/E_target+target/2*sp.sparse.eye(mat_1_NKP.shape[0]))
#prec_thesis_test=sparse.kron(inverted_MAT_1,sp.sparse.eye(N_y**2))
#approximated_matrix=prec_thesis_test

#print("diff norm prec")
#print(sp.sparse.linalg.norm(approximated_matrix-L_mat_NKP))
#def NKP_Prec(x,*args):
 #   x=np.reshape(x,(-1,1))
  #  return approximated_matrix@x


#####################################
for i in range(num_tests):
    start_time=time.time()
    # eigenval,eigenvec=jdqr.jdqr(Hamiltonian_TISE/E_target,num=1,target=-target,tol=1e-10,return_eigenvectors=True,subspace_dimensions=(20,50),maxit=2500,prec=test_2d_prec_2)
    # eigenval,eigenvec=jdqr.jdqr(Hamiltonian_TISE/E_target,num=1,target=-target,tol=1e-10,return_eigenvectors=True,subspace_dimensions=(20,50),maxit=2500)
    eigenval,eigenvec=jdqr.jdqr(Hamiltonian_TISE/E_target,num=1,target=-target,tol=1e-10,return_eigenvectors=True,subspace_dimensions=(20,50),maxit=2500,prec=prec_2d       )

    end_time=time.time()
    time_arr.append(end_time-start_time)

print("elapsed time:", np.mean(time_arr))