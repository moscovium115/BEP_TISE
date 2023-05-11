import jadapy
import numpy as np
from jadapy import jdqr
import scipy as sp
import time

#3x3 matrix
A=np.array([[1.0,2.0,3.0],[3.0,2.0,1.0],[2.0,1.0,3.0]])
print(A)
#wanneer num=3, want de matrix kan uberhaupt niet meer dan 3 versch. eigenwaarde hebben, geeft dit geen zero division error
eigenval,eigenvec=jdqr.jdqr(A,num=3,return_eigenvectors=True)

#dit geeft een zero division error, jdqr() berekent alle eigenwaarde indien num=... niet gespecificeerd is
eigenval,eigenvec=jdqr.jdqr(A,return_eigenvectors=True)

#Zo gebruik je volgens mij een preconditioner, comment de regel hierboven die een error raist, zodat je de onderste regel kunt runnen
inv = sp.sparse.linalg.spilu(sp.sparse.csc_matrix(A))
def _prec(x, *args):
    return inv.solve(x)

eigenval,eigenvec=jdqr.jdqr(A,num=3,return_eigenvectors=True,prec=_prec)

#In jadapy zit in het bestand "tests\test_jdqr.py" een voorbeeld van hoe je een preconditioner gebruikt.









