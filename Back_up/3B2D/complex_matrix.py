from jadapy import jdqr
from jadapy import Target
import numpy as np


A=np.array([[2,1j],[0,1]])
eigenval,eigenvec=jdqr.jdqr(A,num=1,target=1,tol=1e-10,return_eigenvectors=True,arithmetic="complex",subspace_dimensions=(20,50),maxit=2500)
print(eigenvec[:,0])