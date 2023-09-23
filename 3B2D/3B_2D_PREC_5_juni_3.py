import numpy as np


#twee matrices
F=np.array([[1,2,3,4],[4,5,6,8],[7,8,10,11],[12,13,14,15]])
E=np.array([[10,11,12],[13,14,15],[16,17,20]])

#een vector
Right_vector=np.arange(F.shape[0]*E.shape[0])
Right_vector=np.reshape(Right_vector,(-1,1))

def solve(F,E,Right_vector):
    """Solves (F (x) E ) y=x for y, where (x) is the Kronecker product."""

    Right_vector_matrix=np.reshape(Right_vector,(E.shape[0],F.shape[0]),order='F')

    G=np.zeros((E.shape[0],F.shape[0]))

    for i in range(Right_vector_matrix.shape[1]):
        g_i=np.linalg.solve(E,Right_vector_matrix[:,i])
        G[:,i]=g_i

    Vectorisation_mat=np.zeros((F.shape[0],E.shape[0]))

    for i in range(Right_vector_matrix.shape[0]):
        g_i_t=np.linalg.solve(F,G[i,:])
        Vectorisation_mat[:,i]=g_i_t

    #van matrix naar vector gaan, voert al een transpose uit, tenzij je "order=F" meegeeft
    vector=np.reshape(Vectorisation_mat.T,(-1,1),order='F')
    Left_mat=np.kron(F,E)
    print("actual solution:")
    exact_sol=np.linalg.solve(Left_mat,Right_vector)
    print(Left_mat@exact_sol)
    print(Left_mat@vector)
    return vector


#de functie hierboven returnt niet helemaal de juiste preconditioner. Je wilt  (A-target *I) x =b oplossen, en die target moet je in de A,B,C,D matrices brengen
print("solv1 vec:",solve(F,E,Right_vector))

N_y=3

from scipy.linalg import solve_sylvester
from scipy.linalg import schur, eigvals

A=np.random.random((N_y,N_y))
B=np.random.random((N_y,N_y))
schur_A= schur(A)
schur_B= schur(B)
Right_vector=np.arange(N_y**2)
Right_vector=np.reshape(Right_vector,(-1,1))
def Sylv_1_d_prec(x,A,B):
    C=np.reshape(x,(A.shape[0],B.shape[0]))
    C=np.asarray(C, dtype=np.float64)
    X = solve_sylvester(A,B, C.T,schur_A,schur_B)
    solution_vector=np.reshape(X.T,(N_y*N_y,1))
    return solution_vector

sol=Sylv_1_d_prec(Right_vector,A,B)

print("Sylv_1_d_prec",sol)

L_mat=np.kron(np.identity(N_y),A)+np.kron(B,np.identity(N_y))
print(L_mat@sol)