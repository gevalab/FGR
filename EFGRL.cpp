/* This code calculates Fermi Golden Rule rate with linear coupling (non-Condon)
    using multiple harmonic oscillator model, under different approximation levels
    (c) Xiang Sun 2015
    EFGRL.cpp calculates exact normal modes, as well as
    EFGRL.cpp uses the J_eff omega=(w+1)*d_omega_eff instead of w*d_omega_eff (as in EFGRL2.cpp)
 */

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
using namespace std;


//For normal cases
const int bldim = 4;
const int eldim = 3;
double beta_list[bldim] = {0.2, 1.0, 2.0, 5.0};//{10};  //{0.1, 1.0, 10.0};  //{0.2, 1.0, 5.0};
double eta_list[eldim] = {0.5, 1.0, 5.0}; //{0.1}; //{0.1, 1.0, 10.0};  //{0.5, 1.0, 5.0};
const int LEN = 1024; //512; //number of t choices 512 for normal case, 1024 for quantum b10e0.1
const double DeltaT = 0.2;//0.2 (LEN=512) or 0.3 (LEN=2014) for ohmic //FFT time step
const int n_omega = 500; //normal cases 200, 500 for b10e0.1

/*
//For quantum limit case
const int bldim = 1;
const int eldim = 1;
double beta_list[bldim] = {5}; //{10};
double eta_list[eldim] = {0.5}; //{0.1};
const int LEN = 1024; //512; //number of t choices 512 for normal case, 1024 for quantum b10e0.1
const double DeltaT = 0.3;//0.2 (LEN=512) or 0.3 (LEN=1024) for ohmic //FFT time step
const int n_omega = 1000; //normal cases 200, 500 for b10e0.1
*/

double beta = 1;//0.2;//1;//5;
double eta = 1; //0.2;//1;//5;
double Omega = 0.5; //primary mode freq
double y_0 = 1; //shift of primary mode
const double DAcoupling = 0.1;

const double omega_max = 15;//10, 15 or 20 for ohmic
const double d_omega = omega_max / n_omega;// ~ 0.1 for ohmic
const double d_omega_eff = omega_max / n_omega; //for effective SD sampling rate

const double T0 = -DeltaT*LEN/2;
const double pi = 3.14159265358979;
const double RT_2PI= sqrt(2*pi);
const double hbar = 1;

void FFT(int dir, int m, double *x, double *y); //Fast Fourier Transform, 2^m data
double S_omega_ohmic(double omega, double eta); //S(omega) spectral density
double J_omega_ohmic(double omega, double eta);//J(omega) bath Ohmic SD
double J_omega_ohmic_eff(double omega, double eta); //J effective SD for Ohmic bath
void Integrand_exact(double omega, double t, double &re, double &im);
void Integrand_LSC(double omega, double t, double &re, double &im);
void Integrand_CAV(double omega, double t, double &re, double &im);
void Integrand_CD(double omega, double t, double &re, double &im);
void Integrand_W0(double omega, double t, double &re, double &im);
void Integrand_Marcus(double omega, double t, double &re, double &im);
void Linear_exact(double omega, double t, double req, double &re, double &im);
void Linear_LSC(double omega, double t, double req, double &re, double &im);
void Linear_LSC_PW(double omega, double t, double req, double &re, double &im);
void Linear_CAV(double omega, double t, double req, double &re, double &im);
void Linear_CD(double omega, double t, double req, double &re, double &im);
void Linear_W0(double omega, double t, double req, double &re, double &im);
void Linear_Marcus(double omega, double t, double req, double &re, double &im);
double Integrate(double *data, int n, double dx);
double Sum(double *data, int n);
double** Create_matrix(int row, int col);


extern "C" {
    void dsyev_(const char &JOBZ, const char &UPLO, const int &N, double *A,
                const int &LDA, double *W, double *WORK, const int &LWORK,
                int &INFO);
}


int main (int argc, char *argv[]) {
    
    stringstream ss;
    string emptystr("");
    string nameapp("");
    string filename;
    string idstr("");

    int mm(0), nn(1); // nn = 2^mm is number of (complex) data to FFT
	
	while (nn < LEN ) {
		mm++;
		nn *= 2;
	} //nn is the first 2^m that larger LEN
	
	double *corr1 = new double [nn];
	double *corr2 = new double [nn];
    
    double *corr1_orig = new double [nn]; //shifted origin to T0
    double *corr2_orig = new double [nn];
    
    double t;
    int i, j, a, b;
    double omega;
    int w; //count of omega
    double integ_re[n_omega];
    double integ_im[n_omega];
    
    ofstream outfile;
    
    double integral_re, integral_im;
    integral_re = integral_im = 0;
    double *integrand = new double [n_omega];
    double Er=0;
    double a_parameter=0;
    double SD[n_omega];
    double J_eff[n_omega];

    double shift = T0 / DeltaT;
    double N = nn;
    double linear_accum_re;
    double linear_accum_im;
    double linear_re;
    double linear_im;
    double temp_re;
    double temp_im;
    double req_eff[n_omega];//req of effective SD

    //Dimension of matrix (Check this every time)
    int dim = n_omega;
    //-------- initiate LAPACK -----------
    int col = dim, row = dim;
    //Allocate memory for the eigenvlues
    double *eig_val = new double [row];
    //Allocate memory for the matrix
    double **matrix = new double* [col];
    matrix[0] = new double [col*row];
    for (int i = 1; i < col; ++i)
        matrix[i] = matrix[i-1] + row;
    //Parameters for dsyev_ in LAPACK
    int lwork = 6*col, info;
    double *work = new double [lwork];
    //-------------------------------------
    double **D_matrix; // the Hessian matrix
    D_matrix = Create_matrix(n_omega, n_omega);
    double **TT_ns;
    TT_ns = Create_matrix(n_omega, n_omega);
    //transformation matrix: [normal mode]=[[TT_ns]]*[system-bath]
    //TT_ns * D_matrix * T_sn = diag, eigenvectors are row-vector of TT_ns
    double omega_nm[n_omega]; //normal mode frequencies
    double req_nm[n_omega]; //req of normal modes (acceptor shift)
    double c_nm[n_omega];//coupling strength of normal modes
    double S_array[n_omega];//Huang-Rhys factor for normal modes
    double c_bath[n_omega]; //secondary bath mode min shift coefficients
    double gamma_array[n_omega]; //linear coupling coefficient

    int beta_index(0);
    int eta_index(0);
    
    cout << "--------- EFGRL in Condon case --------" << endl;
    
    //BEGIN loop through thermal conditions
    int case_count(0);
    for (beta_index = 0; beta_index < bldim; beta_index++)
        for (eta_index = 0; eta_index < eldim; eta_index++)
    {
        beta = beta_list[beta_index];
        eta = eta_list[eta_index];
        ss.str("");
        nameapp = "";
        ss << "b" << beta;
        ss << "e" << eta;
        nameapp = ss.str();

            
    
    //------- setting up spectral density ----------
    for (w = 0; w < n_omega; w++) SD[w] = S_omega_ohmic(w*d_omega, eta);
    for (w = 0; w < n_omega; w++) J_eff[w] = J_omega_ohmic_eff((w+1)*d_omega_eff, eta);
    for (w = 0; w < n_omega; w++) req_eff[w] = sqrt(8 * hbar * J_eff[w] / (pi * (w+1) * d_omega_eff* (w+1) * d_omega_eff*(w+1)));//eq min for each Jeff normal mode
    
    for (w = 1; w < n_omega; w++) {//ohmic bath mode coupling, essential to exact discrete modes
        c_bath[w] = sqrt( 2 / pi * J_omega_ohmic(w*d_omega, eta) * d_omega * d_omega * w);
    }
    
    
    //********** BEGIN of Normal mode analysis ***********
    
    for (i=0; i< n_omega; i++) for (j=0; j<n_omega ;j++) D_matrix[i][j] = 0;
    D_matrix[0][0] = Omega*Omega;
    for (w =1 ; w < n_omega ; w++) {
        D_matrix[0][0] += pow(c_bath[w]/(w*d_omega) ,2);
        D_matrix[0][w] = D_matrix[w][0] = c_bath[w];
        D_matrix[w][w] = pow(w*d_omega ,2);
    }

    for (i=0; i < dim; i++)
        for (j=0; j < dim; j++)
            matrix[j][i] = D_matrix[i][j]; //switch i j to match with Fortran array memory index
        
    //diagonalize matrix, the eigenvectors transpose is in result matrix => TT_ns.
    dsyev_('V', 'L', col, matrix[0], col, eig_val, work, lwork, info); //diagonalize matrix
    if (info != 0) cout << "Lapack failed. " << endl;
    
    for (i=0; i < dim; i++) omega_nm[i] = sqrt(eig_val[i]);
    
    for (i=0; i < dim; i++)
        for (j=0; j < dim; j++) TT_ns[i][j] = matrix[i][j];

    // the coefficients of linear electronic coupling in normal modes (gamma[j]=TT_ns[j][0]*gamma_y), here gamma_y=1
    for (i=0; i<n_omega; i++) gamma_array[i] = TT_ns[i][0];
    
    //req of normal modes (acceptor shift)
    for (i=0; i<n_omega; i++) {
        req_nm[i] = 1 * TT_ns[i][0];
        for (a=1; a < n_omega; a++) req_nm[i] -= TT_ns[i][a] * c_bath[a] / (a*d_omega * a*d_omega);
    }
    for (i=0; i<n_omega; i++) {
        //tilde c_j coupling strength normal mode
        c_nm[i] = req_nm[i] * omega_nm[i] * omega_nm[i];
        req_nm[i] *= 2 * y_0;
        //discrete Huang-Rhys factor
        S_array[i] = omega_nm[i] * req_nm[i] * req_nm[i] / 2.0;
    }
    //******** END of Normal mode analysis **************
    
    //exact reorganization energy Er for Marcus theory from normal modes
    Er = 0;
    a_parameter = 0;
    //for (i = 0; i < n_omega; i++) Er += 2.0 * c_nm[i] * c_nm[i] / (omega_nm[i] * omega_nm[i]);
    for (i = 0; i < n_omega; i++) Er += 0.5 * omega_nm[i] * omega_nm[i] * req_nm[i] * req_nm[i]; //S_array[i] * omega_nm[i];
    for (i = 0; i < n_omega; i++) a_parameter += 0.5 * S_array[i] * omega_nm[i] * omega_nm[i] /tanh(beta*hbar* omega_nm[i] *0.5);
    cout << "Er (normal mode) = " << Er << endl;
    cout << "a_parameter (normal mode) = "<< a_parameter << endl;
    
    /*
    //test cases: [1] Eq FGR Condon using continuous SD J_eff(\omega)
    //Exact or LSC approximation
    for (i = 0; i < nn; i++) corr1[i] = corr2[i] = 0; //zero padding
    for (i = 0; i < LEN; i++) {
        t = T0 + DeltaT * i;
        integ_re[0] =0;
        integ_im[0] =0;
        for (w = 0; w < n_omega; w++) {
            omega = (w+1) * d_omega_eff;
            Integrand_LSC(omega, t, integ_re[w], integ_im[w]);
            integ_re[w] *= 4*J_eff[w]/pi/(omega*omega);
            integ_im[w] *= 4*J_eff[w]/pi/(omega*omega);
        }
        integral_re = Integrate(integ_re, n_omega, d_omega);
        integral_im = Integrate(integ_im, n_omega, d_omega);
        
        corr1[i] = exp(-1 * integral_re) * cos(integral_im);
        corr2[i] = -1 * exp(-1 * integral_re) * sin(integral_im);
    }
    
    FFT(-1, mm, corr1, corr2);//notice its inverse FT
    
    for(i=0; i<nn; i++) { //shift time origin
        corr1_orig[i] = corr1[i] * cos(2*pi*i*shift/N) - corr2[i] * sin(-2*pi*i*shift/N);
        corr2_orig[i] = corr2[i] * cos(2*pi*i*shift/N) + corr1[i] * sin(-2*pi*i*shift/N);
    }
   
    outfile.open("Exact_EFGR_Jeff.dat");
    for (i=0; i<nn/2; i++) outfile << corr1_orig[i]*LEN*DeltaT*DAcoupling*DAcoupling << endl;
    outfile.close();
    outfile.clear();
     */

    //test cases: [2] Eq FGR Condon using discreitzed SD J_o(\omega)
    for (i = 0; i < nn; i++) corr1[i] = corr2[i] = 0; //zero padding
    for (i = 0; i < LEN; i++) {
        t = T0 + DeltaT * i;
        integ_re[0] = 0;
        integ_im[0] = 0;
        for (w = 0; w < n_omega; w++) {
            Integrand_exact(omega_nm[w], t, integ_re[w], integ_im[w]);
            integ_re[w] *= S_array[w];
            integ_im[w] *= S_array[w];
        }
        integral_re = Sum(integ_re, n_omega);
        integral_im = Sum(integ_im, n_omega);
        corr1[i] = exp(-1 * integral_re) * cos(integral_im);
        corr2[i] = -1 * exp(-1 * integral_re) * sin(integral_im);
    }
    
    FFT(-1, mm, corr1, corr2);//notice its inverse FT
    
    for(i=0; i<nn; i++) { //shift time origin
        corr1_orig[i] = corr1[i] * cos(2*pi*i*shift/N) - corr2[i] * sin(-2*pi*i*shift/N);
        corr2_orig[i] = corr2[i] * cos(2*pi*i*shift/N) + corr1[i] * sin(-2*pi*i*shift/N);
    }
    
    outfile.open((emptystr + "Exact_EFGR_" + nameapp + ".dat").c_str());
    for (i=0; i<nn/2; i++) outfile << corr1_orig[i]*LEN*DeltaT*DAcoupling*DAcoupling << endl;
    outfile.close();
    outfile.clear();
    

    

    //[A] exact quantum EFGR Linear coupling with discrete normal modes
    for (i = 0; i < nn; i++) corr1[i] = corr2[i] = 0;
    for (i = 0; i < LEN; i++) {
        t = T0 + DeltaT * i;
        integ_re[0] = 0;
        integ_im[0] = 0;
        linear_accum_re = 0;
        linear_accum_im = 0;
        for (w = 0; w < n_omega; w++) {
            Integrand_exact(omega_nm[w], t, integ_re[w], integ_im[w]);
            integ_re[w] *= S_array[w];
            integ_im[w] *= S_array[w];
            
            Linear_exact(omega_nm[w], t, req_nm[w], linear_re, linear_im);
            linear_accum_re += linear_re * gamma_array[w] * gamma_array[w];
            linear_accum_im += linear_im * gamma_array[w] * gamma_array[w];
        }
        integral_re = Sum(integ_re, n_omega);
        integral_im = Sum(integ_im, n_omega);
        temp_re = exp(-1 * integral_re) * cos(integral_im);
        temp_im = -1 * exp(-1 * integral_re) * sin(integral_im);
        corr1[i] = temp_re * linear_accum_re - temp_im * linear_accum_im;
        corr2[i] = temp_re * linear_accum_im + temp_im * linear_accum_re;
    }
    
    FFT(-1, mm, corr1, corr2);//notice its inverse FT
    
    for(i=0; i<nn; i++) { //shift time origin
        corr1_orig[i] = corr1[i] * cos(2*pi*i*shift/N) - corr2[i] * sin(-2*pi*i*shift/N);
        corr2_orig[i] = corr2[i] * cos(2*pi*i*shift/N) + corr1[i] * sin(-2*pi*i*shift/N);
    }
    
    outfile.open((emptystr + "Exact_EFGRL_" + nameapp + ".dat").c_str());
    for (i=0; i<nn/2; i++) outfile << corr1_orig[i]*LEN*DeltaT*DAcoupling*DAcoupling << endl;
    outfile.close();
    outfile.clear();

    //[B] LSC approximation using normal modes
    for (i = 0; i < nn; i++) corr1[i] = corr2[i] = 0;
    for (i = 0; i < LEN; i++) {
        t = T0 + DeltaT * i;
        integ_re[0] = 0;
        integ_im[0] = 0;
        linear_accum_re = 0;
        linear_accum_im = 0;
        for (w = 0; w < n_omega; w++) {
            Integrand_LSC(omega_nm[w], t, integ_re[w], integ_im[w]);
            integ_re[w] *= S_array[w];
            integ_im[w] *= S_array[w];
            
            Linear_LSC(omega_nm[w], t, req_nm[w], linear_re, linear_im);
            linear_accum_re += linear_re * gamma_array[w] * gamma_array[w];
            linear_accum_im += linear_im * gamma_array[w] * gamma_array[w];
        }
        integral_re = Sum(integ_re, n_omega);
        integral_im = Sum(integ_im, n_omega);
        temp_re = exp(-1 * integral_re) * cos(integral_im);
        temp_im = -1 * exp(-1 * integral_re) * sin(integral_im);
        corr1[i] = temp_re * linear_accum_re - temp_im * linear_accum_im;
        corr2[i] = temp_re * linear_accum_im + temp_im * linear_accum_re;
    }
    
    FFT(-1, mm, corr1, corr2);//notice its inverse FT
    
    for(i=0; i<nn; i++) { //shift time origin
        corr1_orig[i] = corr1[i] * cos(2*pi*i*shift/N) - corr2[i] * sin(-2*pi*i*shift/N);
        corr2_orig[i] = corr2[i] * cos(2*pi*i*shift/N) + corr1[i] * sin(-2*pi*i*shift/N);
    }
    
    outfile.open((emptystr + "LSC_EFGRL_" + nameapp + ".dat").c_str());
    for (i=0; i<nn/2; i++) outfile << corr1_orig[i]*LEN*DeltaT*DAcoupling*DAcoupling << endl;
    outfile.close();
    outfile.clear();
        
    //[Bb] LSC approximation using normal modes with product of Wigner (less accurate)
    for (i = 0; i < nn; i++) corr1[i] = corr2[i] = 0;
    for (i = 0; i < LEN; i++) {
        t = T0 + DeltaT * i;
        integ_re[0] = 0;
        integ_im[0] = 0;
        linear_accum_re = 0;
        linear_accum_im = 0;
        for (w = 0; w < n_omega; w++) {
            Integrand_LSC(omega_nm[w], t, integ_re[w], integ_im[w]);
            integ_re[w] *= S_array[w];
            integ_im[w] *= S_array[w];
            
            Linear_LSC_PW(omega_nm[w], t, req_nm[w], linear_re, linear_im);
            linear_accum_re += linear_re * gamma_array[w] * gamma_array[w];
            linear_accum_im += linear_im * gamma_array[w] * gamma_array[w];
        }
        integral_re = Sum(integ_re, n_omega);
        integral_im = Sum(integ_im, n_omega);
        temp_re = exp(-1 * integral_re) * cos(integral_im);
        temp_im = -1 * exp(-1 * integral_re) * sin(integral_im);
        corr1[i] = temp_re * linear_accum_re - temp_im * linear_accum_im;
        corr2[i] = temp_re * linear_accum_im + temp_im * linear_accum_re;
    }
    
    FFT(-1, mm, corr1, corr2);//notice its inverse FT
    
    for(i=0; i<nn; i++) { //shift time origin
        corr1_orig[i] = corr1[i] * cos(2*pi*i*shift/N) - corr2[i] * sin(-2*pi*i*shift/N);
        corr2_orig[i] = corr2[i] * cos(2*pi*i*shift/N) + corr1[i] * sin(-2*pi*i*shift/N);
    }
    
    outfile.open((emptystr + "LSCPW_EFGRL_" + nameapp + ".dat").c_str());
    for (i=0; i<nn/2; i++) outfile << corr1_orig[i]*LEN*DeltaT*DAcoupling*DAcoupling << endl;
    outfile.close();
    outfile.clear();
    
    
    //[C] C-AV approximation using normal modes
    for (i = 0; i < nn; i++) corr1[i] = corr2[i] = 0;
    for (i = 0; i < LEN; i++) {
        t = T0 + DeltaT * i;
        integ_re[0] = 0;
        integ_im[0] = 0;
        linear_accum_re = 0;
        linear_accum_im = 0;
        for (w = 0; w < n_omega; w++) {
            Integrand_CAV(omega_nm[w], t, integ_re[w], integ_im[w]);
            integ_re[w] *= S_array[w];
            integ_im[w] *= S_array[w];
            
            Linear_CAV(omega_nm[w], t, req_nm[w], linear_re, linear_im);
            linear_accum_re += linear_re * gamma_array[w] * gamma_array[w];
            linear_accum_im += linear_im * gamma_array[w] * gamma_array[w];
        }
        integral_re = Sum(integ_re, n_omega);
        integral_im = Sum(integ_im, n_omega);
        temp_re = exp(-1 * integral_re) * cos(integral_im);
        temp_im = -1 * exp(-1 * integral_re) * sin(integral_im);
        corr1[i] = temp_re * linear_accum_re - temp_im * linear_accum_im;
        corr2[i] = temp_re * linear_accum_im + temp_im * linear_accum_re;
    }
    
    FFT(-1, mm, corr1, corr2);//notice its inverse FT
    
    for(i=0; i<nn; i++) { //shift time origin
        corr1_orig[i] = corr1[i] * cos(2*pi*i*shift/N) - corr2[i] * sin(-2*pi*i*shift/N);
        corr2_orig[i] = corr2[i] * cos(2*pi*i*shift/N) + corr1[i] * sin(-2*pi*i*shift/N);
    }
    
    outfile.open((emptystr + "CAV_EFGRL_" + nameapp + ".dat").c_str());
    for (i=0; i<nn/2; i++) outfile << corr1_orig[i]*LEN*DeltaT*DAcoupling*DAcoupling << endl;
    outfile.close();
    outfile.clear();
        
    //[D] C-D approximation using normal modes
    for (i = 0; i < nn; i++) corr1[i] = corr2[i] = 0;
    for (i = 0; i < LEN; i++) {
        t = T0 + DeltaT * i;
        integ_re[0] = 0;
        integ_im[0] = 0;
        linear_accum_re = 0;
        linear_accum_im = 0;
        for (w = 0; w < n_omega; w++) {
            Integrand_CD(omega_nm[w], t, integ_re[w], integ_im[w]);
            integ_re[w] *= S_array[w];
            integ_im[w] *= S_array[w];
            
            Linear_CD(omega_nm[w], t, req_nm[w], linear_re, linear_im);
            linear_accum_re += linear_re * gamma_array[w] * gamma_array[w];
            linear_accum_im += linear_im * gamma_array[w] * gamma_array[w];
        }
        integral_re = Sum(integ_re, n_omega);
        integral_im = Sum(integ_im, n_omega);
        temp_re = exp(-1 * integral_re) * cos(integral_im);
        temp_im = -1 * exp(-1 * integral_re) * sin(integral_im);
        corr1[i] = temp_re * linear_accum_re - temp_im * linear_accum_im;
        corr2[i] = temp_re * linear_accum_im + temp_im * linear_accum_re;
    }
    
    FFT(-1, mm, corr1, corr2);//notice its inverse FT
    
    for(i=0; i<nn; i++) { //shift time origin
        corr1_orig[i] = corr1[i] * cos(2*pi*i*shift/N) - corr2[i] * sin(-2*pi*i*shift/N);
        corr2_orig[i] = corr2[i] * cos(2*pi*i*shift/N) + corr1[i] * sin(-2*pi*i*shift/N);
    }
    
    outfile.open((emptystr + "CD_EFGRL_" + nameapp + ".dat").c_str());
    for (i=0; i<nn/2; i++) outfile << corr1_orig[i]*LEN*DeltaT*DAcoupling*DAcoupling << endl;
    outfile.close();
    outfile.clear();
        
    //[E] W-0 (inhomogeneous) approximation using normal modes
    for (i = 0; i < nn; i++) corr1[i] = corr2[i] = 0;
    for (i = 0; i < LEN; i++) {
        t = T0 + DeltaT * i;
        integ_re[0] = 0;
        integ_im[0] = 0;
        linear_accum_re = 0;
        linear_accum_im = 0;
        for (w = 0; w < n_omega; w++) {
            Integrand_W0(omega_nm[w], t, integ_re[w], integ_im[w]);
            integ_re[w] *= S_array[w];
            integ_im[w] *= S_array[w];
            
            Linear_W0(omega_nm[w], t, req_nm[w], linear_re, linear_im);
            linear_accum_re += linear_re * gamma_array[w] * gamma_array[w];
            linear_accum_im += linear_im * gamma_array[w] * gamma_array[w];
        }
        integral_re = Sum(integ_re, n_omega);
        integral_im = Sum(integ_im, n_omega);
        temp_re = exp(-1 * integral_re) * cos(integral_im);
        temp_im = -1 * exp(-1 * integral_re) * sin(integral_im);
        corr1[i] = temp_re * linear_accum_re - temp_im * linear_accum_im;
        corr2[i] = temp_re * linear_accum_im + temp_im * linear_accum_re;
    }
    
    FFT(-1, mm, corr1, corr2);//notice its inverse FT
    
    for(i=0; i<nn; i++) { //shift time origin
        corr1_orig[i] = corr1[i] * cos(2*pi*i*shift/N) - corr2[i] * sin(-2*pi*i*shift/N);
        corr2_orig[i] = corr2[i] * cos(2*pi*i*shift/N) + corr1[i] * sin(-2*pi*i*shift/N);
    }
    
    outfile.open((emptystr + "W0_EFGRL_" + nameapp + ".dat").c_str());
    for (i=0; i<nn/2; i++) outfile << corr1_orig[i]*LEN*DeltaT*DAcoupling*DAcoupling << endl;
    outfile.close();
    outfile.clear();
        
        
    //[F] Marcus-limit approximation using normal modes
    for (i = 0; i < nn; i++) corr1[i] = corr2[i] = 0;
    for (i = 0; i < LEN; i++) {
        t = T0 + DeltaT * i;
        integ_re[0] = 0;
        integ_im[0] = 0;
        linear_accum_re = 0;
        linear_accum_im = 0;
        for (w = 0; w < n_omega; w++) {
            Integrand_Marcus(omega_nm[w], t, integ_re[w], integ_im[w]);
            integ_re[w] *= S_array[w];
            integ_im[w] *= S_array[w];
            
            Linear_Marcus(omega_nm[w], t, req_nm[w], linear_re, linear_im);
            linear_accum_re += linear_re * gamma_array[w] * gamma_array[w];
            linear_accum_im += linear_im * gamma_array[w] * gamma_array[w];
        }
        integral_re = Sum(integ_re, n_omega);
        integral_im = Sum(integ_im, n_omega);
        temp_re = exp(-1 * integral_re) * cos(integral_im);
        temp_im = -1 * exp(-1 * integral_re) * sin(integral_im);
        corr1[i] = temp_re * linear_accum_re - temp_im * linear_accum_im;
        corr2[i] = temp_re * linear_accum_im + temp_im * linear_accum_re;
    }
    
    FFT(-1, mm, corr1, corr2);//notice its inverse FT
    
    for(i=0; i<nn; i++) { //shift time origin
        corr1_orig[i] = corr1[i] * cos(2*pi*i*shift/N) - corr2[i] * sin(-2*pi*i*shift/N);
        corr2_orig[i] = corr2[i] * cos(2*pi*i*shift/N) + corr1[i] * sin(-2*pi*i*shift/N);
    }
    
    outfile.open((emptystr + "Marcus_EFGRL_" + nameapp + ".dat").c_str());
    for (i=0; i<nn/2; i++) outfile << corr1_orig[i]*LEN*DeltaT*DAcoupling*DAcoupling << endl;
    outfile.close();
    outfile.clear();
        
    

    
    /*
    //Classical sampling with donor potential dynamics (freq shifting)
    for (i = 0; i < nn; i++) corr1[i] = corr2[i] = 0;
    for (i=0; i< LEN; i++) {
        t = T0 + DeltaT * i;
        integ_re[0]=integ_im[0]=0;
        for (w = 1; w < n_omega; w++) {
            omega = w * d_omega;
            Integrand_CL_donor(omega, t, integ_re[w], integ_im[w]);
            integ_re[w] *= SD[w];
            integ_im[w] *= SD[w];
        }
        integral_re = Integrate(integ_re, n_omega, d_omega);
        //integral_im = Integrate(integ_im, n_omega, d_omega);
        
        corr1[i] = exp(-1 * integral_re) ;
        corr2[i] = 0;
        //corr1[i] = exp(-1 * integral_re) * cos(integral_im);
        //corr2[i] = -1 * exp(-1 * integral_re) * sin(integral_im);
    }
    
    outfile.open("CL_donor_t_re.dat");
    for (i=0; i< LEN; i++) outfile << corr1[i] << endl;
    outfile.close();
    outfile.clear();
    
    outfile.open("CL_donor_t_im.dat");
    for (i=0; i< LEN; i++) outfile << corr2[i] << endl;
    outfile.close();
    outfile.clear();
    
    FFT(-1, mm, corr1, corr2);
    
    for(i=0; i<nn; i++) { //shift time origin
        corr1_orig[i] = corr1[i] * cos(2*pi*i*shift/N) - corr2[i] * sin(-2*pi*i*shift/N);
        corr2_orig[i] = corr2[i] * cos(2*pi*i*shift/N) + corr1[i] * sin(-2*pi*i*shift/N);
    }
    
    //shift freq -omega_0=-Er
    int shift_f(0);
    shift_f = static_cast<int> (Er/(1.0/LEN/DeltaT)/(2*pi)+0.5);
    
    outfile.open("CL_donor_re.dat");
    for (i=nn-shift_f; i<nn; i++) outfile << corr1_orig[i]*LEN*DeltaT*DAcoupling*DAcoupling << endl;
    for (i=0; i<nn-shift_f; i++) outfile << corr1_orig[i]*LEN*DeltaT*DAcoupling*DAcoupling << endl;
    outfile.close();
    outfile.clear();
    
    */
    
    

    
    


    
    case_count++;
    
    //-------------- Summary ----------------
    
    cout << "CASE # " << case_count <<  " done:" << endl;
    cout << "   beta = " << beta << endl;
    cout << "    eta = " << eta << endl;
    cout << "-------------------" << endl;
            
    } //loop of thermal condition
    
    cout << "DeltaT = " << DeltaT << endl;
    cout << "N = " << LEN << endl;
    cout << "df = " << 1.0/LEN/DeltaT << endl;
    cout << "f_max = " << 0.5/DeltaT << endl;
    cout << "beta = " << beta << endl;
    cout << " eta = " << eta << endl;
    
    cout << "Done." << endl;



    // Deallocate memory
    delete [] eig_val;
    delete [] matrix[0];
    delete [] matrix;
    delete [] work;

    return 0;
}



/********* SUBROUTINE *************/


//spectral densities
double S_omega_ohmic(double omega, double eta) {
    // S_omega= sum_j omega_j * Req_j^2 / 2 hbar delta(\omega - \omega_j)
    return eta * omega * exp(-1 * omega);
}

double J_omega_ohmic(double omega, double eta) {
    //notice definition J(omega) is different from S(omega)
    //J_omega = pi/2 * sum_a c_a^2 / omega_a delta(omega - omega_a)
    return eta * omega * exp(-1 * omega);
}

double J_omega_ohmic_eff(double omega, double eta) {
    //(normal mode) effective SD for Ohmic bath DOF
    //J_omega = pi/2 * sum_a c_a^2 / omega_a delta(omega - omega_a)
    return eta * omega * pow(Omega,4) / ( pow(Omega*Omega - omega*omega, 2) + eta*eta*omega*omega);
}


//FGR exponent integrand and linear term

void Integrand_exact(double omega, double t, double &re, double &im) {
    double Coth = 1.0/tanh(beta*hbar*omega*0.5);
    re = (1-cos(omega*t))*Coth;
    im = sin(omega*t);
    return;
}

void Linear_exact(double omega, double t, double req, double &re, double &im) {
    double Coth = 1.0/tanh(beta*hbar*omega*0.5);
    re = 0.5*hbar/omega*Coth*cos(omega*t) + 0.25*req*req*((1-cos(omega*t))*(1-cos(omega*t)) - sin(omega*t)*sin(omega*t)*Coth*Coth);
    im = -0.5*hbar/omega*sin(omega*t) + 0.5*req*req*Coth*(1-cos(omega*t))*sin(omega*t);
    return;
}

void Integrand_LSC(double omega, double t, double &re, double &im) {
    double Coth = 1.0/tanh(beta*hbar*omega*0.5);
    re = (1-cos(omega*t))*Coth;
    im = sin(omega*t);
    return;
}

void Linear_LSC(double omega, double t, double req, double &re, double &im) {
    //accurate LSC: using Wigner of product
    double Coth = 1.0/tanh(beta*hbar*omega*0.5);
    re = 0.5*hbar/omega* Coth *cos(omega*t) - 0.25*req*req* sin(omega*t)*sin(omega*t)*Coth*Coth;
    im = -0.5*hbar/omega*sin(omega*t) + 0.25*req*req*Coth*(1-cos(omega*t))*sin(omega*t);
    //im = 0;
    return;
}

void Linear_LSC_PW(double omega, double t, double req, double &re, double &im) {
    //less accurate LSC: using product of Wigner
    double Coth = 1.0/tanh(beta*hbar*omega*0.5);
    re = 0.5*hbar/omega* Coth *cos(omega*t) - 0.25*req*req* sin(omega*t)*sin(omega*t)*Coth*Coth;
    //im = -0.5*hbar/omega*sin(omega*t) + 0.25*req*req*Coth*(1-cos(omega*t))*sin(omega*t);
    im = 0;
    return;
}

void Integrand_CAV(double omega, double t, double &re, double &im) {
    re = (1-cos(omega*t))*2/(beta*hbar*omega);
    im = sin(omega*t);
    return;
}

void Linear_CAV(double omega, double t, double req, double &re, double &im) {
    re = (beta*hbar*hbar*cos(omega*t) - req*req*sin(omega*t)*sin(omega*t)) / (beta*beta * hbar*hbar *omega*omega);
    im = 0;
    return;
}

void Integrand_CD(double omega, double t, double &re, double &im) {
    re = (1-cos(omega*t))*2/(beta*hbar*omega);
    im = omega*t;
    return;
}

void Linear_CD(double omega, double t, double req, double &re, double &im) {
    re = (beta*hbar*hbar*cos(omega*t) - req*req*sin(omega*t)*sin(omega*t)) / (beta*beta * hbar*hbar *omega*omega);
    im = 0;
    return;
}

void Integrand_W0(double omega, double t, double &re, double &im) {
    double Coth = 1.0/tanh(beta*hbar*omega*0.5);
    re = omega*omega*t*t*0.5 * Coth;
    im = omega*t;
    return;
}

void Linear_W0(double omega, double t, double req, double &re, double &im) {
    double Coth = 1.0/tanh(beta*hbar*omega*0.5);
    re = ( 0.5*hbar/omega - 0.25*req*req*omega*omega*t*t*Coth)*Coth;
    im = 0;
    return;
}


void Integrand_Marcus(double omega, double t, double &re, double &im) {
    re = omega*t*t/(beta*hbar);
    im = omega*t;
    return;
}
                                                        
void Linear_Marcus(double omega, double t, double req, double &re, double &im) {
  re = 1.0/(beta*omega*omega) - req*req*t*t/(beta*beta*hbar*hbar);
  im = 0;
  return;
}
                                                          
double Integrate(double *data, int n, double dx){
    double I =0;
    I += (data[0]+data[n-1]);//  /2;
    for (int i=1; i< n-1; i++) {
        I += data[i];
    }
    I *= dx;
    return I;
}

double Sum(double *data, int n){
    double I = 0;
    for (int i=0; i< n; i++) {
        I += data[i];
    }
    return I;
}


void FFT(int dir, int m, double *x, double *y)
    {/*
      This code computes an in-place complex-to-complex FFT Written by Paul Bourke
      x and y are the real and imaginary arrays of N=2^m points.
      dir =  1 gives forward transform
      dir = -1 gives reverse transform
      Formula: forward
                  N-1
                  ---
              1   \           - i 2 pi k n / N
      X(n) = ----  >   x(k) e                       = forward transform
              1   /                                    n=0..N-1
                  ---
                  k=0
      
      Formula: reverse
                  N-1
                  ---
               1  \           i 2 pi k n / N
      X(n) =  ---  >   x(k) e                  = reverse transform
               N  /                               n=0..N-1
                  ---
                  k=0
      */
        
        int n,i,i1,j,k,i2,l,l1,l2;
        double c1,c2,tx,ty,t1,t2,u1,u2,z;
        
        // Calculate the number of points
        n = 1;
        for (i=0;i<m;i++)
            n *= 2;
        
        // Do the bit reversal
        i2 = n >> 1; //i2 = (010 ...0)_2,second highest bit of n=(100 ...0)_2
        j = 0; //reversely bit accumulater from the second highest bit, i2.
        for (i=0;i<n-1;i++) {
            if (i < j) {
                tx = x[i]; //swap(i,j)
                ty = y[i];
                x[i] = x[j];
                y[i] = y[j];
                x[j] = tx;
                y[j] = ty;
            }
            //to find the highest non-one bit, k, from the second highest bit
            k = i2;
            while (k <= j) {
                j -= k;
                k >>= 1;
            }
            j += k; //add 1 reversly
        }
        
        // Compute the Radix-2 FFT: Cooley-Tukey Algorithm
        c1 = -1.0; // c1+i*c2 = -1 = c^(i 2Pi/2) = W_2, def W_N^j = e^(i 2j*Pi/N)
        c2 = 0.0;
        l2 = 1;
        for (l=0;l<m;l++) {
            l1 = l2;
            l2 <<= 1;
            u1 = 1.0;
            u2 = 0.0;
            for (j=0;j<l1;j++) {
                for (i=j;i<n;i+=l2) {
                    //Butterfly calculation of x,y[i] and x,y[i1]:
                    //t1+i*t2 =(u1+i*u2)(x[i1]+i*y[i2]) where u1+i*u2=W_N^j=e^(i 2j*Pi/N)
                    i1 = i + l1;
                    t1 = u1 * x[i1] - u2 * y[i1];
                    t2 = u1 * y[i1] + u2 * x[i1];
                    x[i1] = x[i] - t1;
                    y[i1] = y[i] - t2;
                    x[i] += t1;
                    y[i] += t2;
                }
                // i1+i*u2 *= c1+i*c2, or W_N
                z =  u1 * c1 - u2 * c2;
                u2 = u1 * c2 + u2 * c1;
                u1 = z;
            }
            //c1+i*c2 = sqrt(c1+i*c2) eg. W_2 --> W_4 ...
            c2 = sqrt((1.0 - c1) / 2.0);
            if (dir == 1)
                c2 = -c2;
            c1 = sqrt((1.0 + c1) / 2.0);
        }
        
        // times STEPS*DeltaT forward FFT (time --> freq)
        /*if (dir == 1) {
         for (i=0; i<n; i++) {
         x[i] *= 1;//DeltaT;
         y[i] *= 1;//DeltaT;
         }
         }*/
        
        // Scaling for inverse transform
        
        if (dir == -1) {
            for (i=0;i<n;i++) {
                x[i] /= n;
                y[i] /= n;
            }
        }
        
        /*
        //for symmetrical FT, 
        double sqn;
        sqn = sqrt(n);
        for (i=0;i<n;i++) {
            x[i] /= sqn;
            y[i] /= sqn;
        }
        */
        
        return;
    }



double** Create_matrix(int row, int col) {
  double **matrix = new double* [col];
  matrix[0] = new double [col*row];
  for (int i = 1; i < col; ++i)
      matrix[i] = matrix[i-1] + row;
  return matrix;
}


