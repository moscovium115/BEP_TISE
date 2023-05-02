#include <iostream>

#include <cmath>

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}

extern "C" long long Init_First_derivative_matrix(double* dirac1,double* chebyshev_nodes, int N_matrix){

    //N_matrix is t aantal rijen(en ook kolommen) van de matrix
    //i is de rij, j is de kolom, dus dat is de N*i+j de element van de array omdat de matrix geflattened word
    for (int i=0; i<N_matrix; i++){
        for (int j=0; j<N_matrix; j++) {
            double x_i=chebyshev_nodes[i];
            double x_j=chebyshev_nodes[j];
            if (i==j){
                dirac1[N_matrix*i+j]=0.5    *x_j/(1-x_j*x_j);
            }
            else{
                dirac1[N_matrix*i+j]=cos(M_PI*(i+j))*sqrt((1-x_j*x_j)/(1-x_i*x_i))/(x_i-x_j);
            }
        }

    }


    return 0;
}


extern "C" long long Init_Second_derivative_matrix(double* dirac2,double* chebyshev_nodes, int N_matrix){
    //N_matrix is t aantal rijen(en ook kolommen) van de matrix
    //i is de rij, j is de kolom, dus dat is de N*i+j de element van de array omdat de matrix geflattened word
    for (int i=0; i<N_matrix; i++){
        for (int j=0; j<N_matrix; j++) {
            double x_i=chebyshev_nodes[i];
            double x_j=chebyshev_nodes[j];
            if (i==j){
                dirac2[N_matrix*i+j]=(x_j*x_j)/((1-x_j*x_j)*(1-x_j*x_j))-(N_matrix*N_matrix-1)/(3*(1-x_j*x_j));
            }
            else{
                //cosinus is wss het beste om te verbeteren, bijv k=1, en daarna bij elke iterati k*(-1)
                double term_1=x_i/(1-x_i*x_i)-2/(x_i-x_j);
                double term_2=cos(M_PI*(i+j))*sqrt((1-x_j*x_j)/(1-x_i*x_i))/(x_i-x_j);

                dirac2[N_matrix*i+j]=term_1*term_2;
            }
        }
    }
    return 0;
}

extern "C" double Chebyshev_Polynomial(int n, double x){
    //check of x een chebyshev node is, zo ja, dan is d cheb polynoom 0
//    std::cout<<"cheb pol iteration"<<n<<std::endl;


    if (n==0){
//        std::cout<<"n=0 is niet gedefinieerd"<<std::endl;
        return 1;
    }
    else if (n==1){
        return x;
    }
    else{
        return 2*x*Chebyshev_Polynomial(n-1,x)-Chebyshev_Polynomial(n-2,x);
    }
}

//
extern "C" double Chebyshev_Polynomial_Derivative(int n, double x){
    //check of x een chebyshev node is, zo ja, dan is d cheb polynoom 0

    if (n==0){
//        std::cout<<"n=0 is niet gedefinieerd"<<std::endl;
        return 0;
    }
    else if (n==1){
        return 1;
    }
    else{
        return 2*(Chebyshev_Polynomial(n-1,x)+x*Chebyshev_Polynomial_Derivative(n-1,x))-Chebyshev_Polynomial_Derivative(n-2,x);
    }
}

extern "C" double Rational_Chebyshev(double* cheb_nodes, int n, double x_i, int j){
    std::cout<<"rat_cheb_iteration"<<j<<std::endl;

    double result=Chebyshev_Polynomial(n,x_i)/(Chebyshev_Polynomial_Derivative(n,cheb_nodes[j])*(x_i-cheb_nodes[j]));
    return result;
}

//extern "C" double Rational_Chebyshev(double* cheb_nodes, int n, double x_i, int j){
//    std::cout<<"rat_cheb_iteration "<<j<<std::endl;
//    double result;
//    bool bruh_condition=false;
//    for (int test_index=0; test_index<n; test_index++){
//        if (cheb_nodes[test_index]==x_i){
//            bruh_condition=true;
//            break;
//
//        }
//    }
//    if (bruh_condition==true){
//        result = 0;
//    }
//    else{
//        result=Chebyshev_Polynomial(n,x_i)/(Chebyshev_Polynomial_Derivative(n,cheb_nodes[j])*(x_i-cheb_nodes[j]));
//    }
//    return result;
//}

extern "C" long long Approximation_Chebyshev(double* weights_TISE, double* func, double* cheb_nodes, double* x,int len_x_arr,int n){
    for (int i=0; i<len_x_arr; i++){
        for (int j=0; j<n; j++){
            func[i]+=weights_TISE[j]*Rational_Chebyshev(cheb_nodes,n,x[i], j);
        }
    }
    return 0;
}
