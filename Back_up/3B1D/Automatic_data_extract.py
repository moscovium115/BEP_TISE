import numpy as np
from efficient_multiple_kronecker_vec_multiplication import *
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
from scipy.sparse.linalg import LinearOperator


# energy_levels=[1,2,3]
# N_x_list=[64,128,256,512]
energy_levels=[3]
N_x_list=[256]


# energy_levels=[3]
# N_x_list=[128]

def add_line_to_file(filename, text_to_add):
    try:
        with open(filename, 'a') as file:
            file.write(text_to_add + '\n')
    except IOError as e:
        print(f"An error occurred: {e}")

# Example usage:
filename = 'gd_test_results.txt'

#target veranderen bij elke loop eigenlijk.

text_to_add="Bartels-Stewart prec"
add_line_to_file(filename, text_to_add)

for energy in energy_levels:
    text_to_add="\n"
    add_line_to_file(filename, text_to_add)
    for N_x in N_x_list:

        M,m=20,1
        mass_ratio=M/m
        alpha_x=-1/((1+mass_ratio))
        alpha_y=(1+2*mass_ratio)/(-4*(1+1*mass_ratio))
        energy_level=energy


        #Select which energy level you want to compute
        if energy_level==1:
            E_target=1e-1
            #checkpoint
            v0=0.344595351
        elif energy_level==2:
            E_target=1e-2
            v0=0.088873721
        else:
            E_target=1e-3
            # v0=0.026134365
            # v0=0.02613436619101265
            v0=0.0261343345

        N_y=int(N_x/2)


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
        result_array_3=1*Test_x[:, np.newaxis] +0* Test_y[np.newaxis, :]

        # Print the result
        rel_distance_1=np.abs(np.ravel(result_array_1))
        rel_distance_2=np.abs(np.ravel(result_array_2))
        rel_distance_3=np.abs(np.ravel(result_array_3))

        #gaussisch
        V_test=-v0*(np.exp(-(rel_distance_1**2))+np.exp(-(rel_distance_2**2))+0*np.exp(-(rel_distance_3**2)))
        V=sp.sparse.diags(V_test)

        #kronecker product optimaliseren
        A_1=alpha_x*D_2_realline_x
        B_1=alpha_y*D_2_realline_y
        from scipy.sparse.linalg import LinearOperator


        mat_vec_count=0
        def mv(v):
            W=np.reshape(v,(N_x,N_y)).T
            test_vec_2=B_1@W+W@A_1.T
            test_vec=v.reshape((N_x*N_y,1))
            test_vec_2=np.reshape(test_vec_2.T,(N_x*N_y,1))+V@test_vec
            global mat_vec_count
            mat_vec_count+=1
            return test_vec_2

        Hamiltonian_TISE= LinearOperator((N_x*N_y,N_x*N_y), matvec=mv)

        target=2.7
        sigma_1=-abs(alpha_y)*target/(abs(alpha_x+alpha_y))
        sigma_2=-abs(alpha_x)*target/(abs(alpha_x+alpha_y))

        #solving sylvester equation
        A=(alpha_y*D_2_realline_y)/E_target-sigma_1*Id_y
        B=(alpha_x*D_2_realline_x)/E_target-sigma_2*Id_x
        B=B.T


        schur_A=schur(A)
        schur_B=schur(B)
        schur_A_tau=[schur_A[0],schur_A[1]]
        schur_B_tau=[schur_B[0],schur_B[1]]
        # print(schur_A[0])

        prec_calls=0
        def prec_func(x, *args):
            #function approximates w in (A-sigma*I)w=x
            # print("debug:", x.shape)
            # print(x)
            # print(N_x*N_y)
            # print(np.real(args[0]))
            est_target=np.real(args[0])
            C=np.reshape(x,(N_x,N_y))
            # schur_A_tau[0]=schur_A[0]+est_target*Id_y*sigma_1/target
            # schur_B_tau[0]=schur_B[0]+est_target*Id_x*sigma_2/target
            # print("debug:",est_target)
            # schur_A_tau[0]=schur_A[0]+abs(alpha_y)*target/(abs(alpha_x+alpha_y))*Id_y
            # schur_B_tau[0]=schur_B[0]+abs(alpha_x)*target/(abs(alpha_x+alpha_y))*Id_x

            X = solve_sylvester(A,B, C.T,schur_A,schur_B)
            result_test=np.reshape(X.T,(N_x*N_y,1))
            global prec_calls
            prec_calls+=1
            # print(schur_A_tau[0])

            return result_test

        num_tests=1

        test_arr=[]
        for test in range(num_tests):
            start_time = time.time()
            # eigenval,eigenvec=jdqr.jdqr(Hamiltonian_TISE/E_target,num=3,target=-target,tol=1e-10,return_eigenvectors=True,arithmetic="complex",subspace_dimensions=(20,50),maxit=2500,prec=prec_func)
            # eigenval,eigenvec,it=jdqr.jdqr(Hamiltonian_TISE/E_target,num=3,target=Target.SmallestRealPart, tol=1e-10,return_eigenvectors=True,arithmetic="complex",subspace_dimensions=(20,50),maxit=2500,prec=prec_func)
            eigenval,eigenvec,it=jdqr.jdqr(Hamiltonian_TISE/E_target,num=3,target=Target.SmallestRealPart, tol=1e-10,return_eigenvectors=True,arithmetic="complex",subspace_dimensions=(20,50),maxit=2500)

            # eigenval,eigenvec=jdqr.jdqr(Hamiltonian_TISE/E_target,num=1,target=-2.71,tol=1e-10,return_eigenvectors=True,subspace_dimensions=(20,50),maxit=2500)
            end_time=time.time()
            test_arr.append(end_time-start_time)

        print(np.mean(test_arr))

        print("Hamiltonian operator applies:", mat_vec_count)

        text_to_add = "energy level: "+str(energy)+" N_x= "+str(N_x)+" time: "+str(np.mean(test_arr))+" hamiltonian calls: "+str(mat_vec_count)+" iterations: "+str(it)+" prec calls: "+str(prec_calls)
        add_line_to_file(filename, text_to_add)
        text_to_add = str(N_x)+"&"+str(N_y)+"&"+str(it)+"&"+str(np.mean(test_arr))+"&"+str(mat_vec_count+2*prec_calls)+"\\"
        add_line_to_file(filename, text_to_add)