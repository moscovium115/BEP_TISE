import numpy as np

def kronecker_sum(A,B):
    return np.kron(A,np.eye(B.shape[0]))+np.kron(np.eye(A.shape[0]),A)


A=np.array([[1,2],[1,3]])

result=kronecker_sum(A,A)

V=[12,1
    ,0,1]
V=np.diag(V)
print(result)
result+=V
print(result)

eigenval, eigenvec=np.linalg.eig(result)
print(eigenval)
print(np.linalg.eigvals(A))

