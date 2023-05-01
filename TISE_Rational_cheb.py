import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sla
from numpy import linalg as LA
from scipy.sparse.linalg import *
import scipy as sp


N=10
chebyshev_nodes=-np.cos(np.pi*(2*np.arange(1,N+1)-1)/(2*N))
print(chebyshev_nodes)
print(len(chebyshev_nodes))


#Recursively constructing Chebyshev polynomials
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
        if (x[i]-chebyshev_nodes[j])==0:
            arr_1[i]=1
        else:
            arr_1[i]=chebyshev_polynomial(n,x[i])/((x[i]-chebyshev_nodes[j])*chebyshev_polynomial_derivative(n,chebyshev_nodes[j]))
    return arr_1


#Linear combination of rational Chebyshev polynomials, weights are found by solving the eigenvalue problem
def chebyshev_approximation_rational(weights,x):
    func=0
    for i in range(len(weights)):
        func+=chebyshev_rational(N,x,i)*weights[i]
    return func


x=chebyshev_nodes

#Construcing second derivative matrix
Dirac_2=np.zeros((N,N))

for i in range(N):
    for j in range(N):
        if i!=j:
            x_i=x[i]
            x_j=x[j]
            term_1=x_i/(1-x_i**2)-2/(x_i-x_j)
            term_2=(-1)**(i+j)  *  np.sqrt((1-x_j**2)/(1-x_i**2))/(x_i-x_j)
            Dirac_2[i,j]=term_1*term_2
        else:
            x_i = x[i]
            x_j = x[j]
            Dirac_2[i,j]=x_j**2/((1-x_j**2)**2)-(N**2-1)/(3*(1-x_j**2))


print(Dirac_2)

#first derivative matrix
Dirac_1=np.zeros((N,N))
for i in range(N):
    for j in range(N):
        x_i = x[i]
        x_j = x[j]
        if i==j:
            Dirac_1[i,j]=1/2*x_j*(1-x_j**2)
        else:
            Dirac_1[i,j]=(-1)**(i+j)*np.sqrt((1-x_j**2)/(1-x_i**2))/(x_i-x_j)



#effective grid size
L=1000000

#constructing A matrix, so that we can extend our domain.

A=np.diag((1/L)*(1-chebyshev_nodes**2)**(3/2))
#B matrix
B=np.diag((-3/L**2)*chebyshev_nodes*(1-chebyshev_nodes**2)**(2))


#Solving ODE: y''+y-2y=-2 on [-1,1] with y(0)=0, y'(0)=0
#Note that the trivial solution does not satisfy the ODE, because of -2 on the right, if the -2 becomes a zero we obtain the trivial solution
Left_matrix_1=Dirac_2+Dirac_1-2*np.identity(N)
Right_vector_1=-2*np.ones(N)
solution_1=np.linalg.solve(Left_matrix_1,Right_vector_1)
x_domain_1=np.linspace(-1,1,100)
plt.plot(x_domain_1,chebyshev_approximation_rational(solution_1,x_domain_1))
plt.show()
#The above solution is correct!




#Solving the Particle in a box situation

#The mapping to extend the solution to the whole real line
mapping=L*chebyshev_nodes/np.sqrt((1-chebyshev_nodes**2))

#constructing potential matrix
#The potential should be infinite outside the box, so we set it to a large number
inf=10000
#width of the box is 0.5
start_x=0
end_x=0.5
pot_arr=inf*np.ones(N)
pot_arr[mapping>=start_x]=0
pot_arr[mapping>=end_x]=inf
# pot_arr=0.5*mapping**2
V=np.diag(pot_arr)


#second derivative matrix for the real line
D_2_realline=A*A*Dirac_2+B*Dirac_1


h_bar=1.0545718e-34
m=9.10938356e-31
alpha=-h_bar**2/(2*m)
Hamiltonian_TISE=alpha*D_2_realline+V


#Weights for the linear combinations are calculated by solving the eigenvalue problem
eigenval, eigenvectors =np.linalg.eig(Hamiltonian_TISE)
print("Eigenvalues:",eigenval)
print("Calculated Energy:",np.min(eigenval),np.min(eigenval.real))
weights_TISE=eigenvectors[:,0]

#Plotting the solution
x_domain_TISE=np.linspace(start_x,end_x,100)
plt.plot(x_domain_TISE,chebyshev_approximation_rational(weights_TISE,x_domain_TISE))
plt.show()

#Solution is totally wrong

