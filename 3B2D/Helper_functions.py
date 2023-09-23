#All functions below are used to construct the dense derivative matrices using C++ code

import ctypes
import numpy as np
import numpy.ctypeslib as ctl


libname="N_body.so"
libdir = './'
mylib=ctl.load_library(libname, libdir)

###Initialize the first derivative matrix function
Initialize_First_Derivative_function=mylib.Init_First_derivative_matrix
Initialize_First_Derivative_function.restype = ctypes.c_longlong
Initialize_First_Derivative_function.argtypes = [ctl.ndpointer(np.float64,flags='aligned, c_contiguous'),ctl.ndpointer(np.float64,flags='aligned, c_contiguous'),ctypes.c_int]

def construct_first_derivative_matrix(cheb_nodes,N):
    Dirac_1=np.zeros((N,N))
    #changes Dirac_1 in place
    Initialize_First_Derivative_function(Dirac_1,cheb_nodes,N)
    return Dirac_1


# N_x=3
# cheb_nodes_x_original=np.cos(np.pi*(2*np.arange(1,2*N_x+1)-1)/(4*N_x))
# N_y=4
# cheb_nodes_y_original=np.cos(np.pi*(2*np.arange(1,2*N_y+1)-1)/(4*N_y))
#
# mat_1=construct_first_derivative_matrix(cheb_nodes_x_original,2*N_x)
#
#
#
# mat_2=construct_first_derivative_matrix(cheb_nodes_y_original,2*N_y)


###Initialize the second derivative matrix
Initialize_Second_Derivative_function=mylib.Init_Second_derivative_matrix
Initialize_Second_Derivative_function.restype = ctypes.c_longlong
Initialize_Second_Derivative_function.argtypes = [ctl.ndpointer(np.float64,flags='aligned, c_contiguous'),ctl.ndpointer(np.float64,flags='aligned, c_contiguous'),ctypes.c_int]

def construct_second_derivative_matrix(cheb_nodes,N):
    Dirac_2=np.zeros((N,N))
    #changes Dirac_2 in place
    Initialize_Second_Derivative_function(Dirac_2,cheb_nodes,N)
    return Dirac_2

