import numpy as np


# Q=np.array([[1,2],[3,4]])
n=5
Q=np.random.rand(n,n)

RHS_vec=np.random.rand(n,1)


def prec(a):
    "preconditioner for Q matrix"
    # print(Q.diagonal())
    # print(a)
    a_prec=np.diag(1/Q.diagonal())@a
    print("a prec:", a_prec)
    return a_prec
def prec_T(a):
    "preconditioner for Q transposed"
    a_prec=np.diag(1/Q.diagonal())@a
    return a_prec
def gradient_descent(RHS_vec):
    e_til=prec(RHS_vec)
    x_k=np.zeros([n,1])
    iterations=1000
    #Nu preconditioner in verwerken
    for i in range(iterations):
        residual=e_til-prec(Q@x_k)
        tau=np.linalg.norm(Q.T@prec_T(residual),ord="fro")**2/np.linalg.norm(prec(Q@Q.T@prec_T(residual)),ord="fro")**2
        x_k=x_k+tau*Q.T@prec_T(residual)
        print("residual:",np.linalg.norm(residual))
    return x_k

x_k=gradient_descent(RHS_vec)
print(Q@x_k)
print(RHS_vec)
