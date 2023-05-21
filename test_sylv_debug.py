import numpy as np
import scipy as sp

C_1=np.array([[1,2,3],[4,5,6],[7,8,9]])
C_2=np.array([[10,11,12],[13,14,15],[16,17,18]])
op_matrix=sp.kron(C_1,np.identity(3))+sp.kron(np.identity(3),C_2)
rhs=np.ones((op_matrix.shape[0],1))
sigma=5

print(op_matrix)

left_matrix=op_matrix-sigma*sp.sparse.identity(op_matrix.shape[0])
sol_1=np.linalg.solve(left_matrix,rhs)
print(sol_1)

from scipy.linalg import solve_sylvester

A=C_2-2.5*np.identity(C_2.shape[0])
B=C_1-2.5*np.identity(C_1.shape[0])
B=B.T
C=np.reshape(rhs,(3,3))
sol_mat=solve_sylvester(A,B,C)
sol_2=np.reshape(sol_mat.T,(9,1))
print(sol_2)
print("proc verschil")
print(100*(sol_1-sol_2)/sol_2)
print(np.max(100*np.abs(sol_1-sol_2)/sol_2))

A=C_2-2.5*np.identity(C_2.shape[0])
A=np.asarray(A,dtype=np.float128)
B=C_1-2.5*np.identity(C_1.shape[0])
B=B.T
B=np.asarray(B,dtype=np.float128)
C=np.reshape(rhs,(3,3))
C=np.asarray(C,dtype=np.float128)
sol_mat=solve_sylvester(A,B,C)
sol_2=np.reshape(sol_mat.T,(9,1))
print("verschil matrix:", A@sol_mat+sol_mat@B-C)

print("betere resolutie:")
print(sol_1-sol_2)
print(np.max(100*np.abs(sol_1-sol_2)/sol_2))


