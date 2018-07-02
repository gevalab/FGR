/* This code calculates Non-equilibrium Fermi Golden Rule rate 
   in non-Condon case using Brownian oscillator model
   compare with linearized semiclassical methods  
   To compile: g++ -o NEFGRL NEFGRL.cpp -llapack -lrefblas -lgfortran
   (c) Xiang Sun 2015
 */

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
using namespace std;

const int bldim = 3;//3;
const int eldim = 3;
double beta_list[bldim] = {1, 2, 5}; //{1, 2, 5};//{0.2, 1.0, 5.0};
double eta_list[eldim] = {0.5, 1, 2};//{0.5, 1, 5};//{0.2, 1.0, 5.0};
double omega_DA_fix = 0; //fixed omega_DA, with scan tp
double s = 5; //Noneq. initial shift of parimary mode
const int n_omega = 500;
const double omega_max = 15;//20;//2.5 for gaussian// 20 for ohmic
const double tp_max = 40; //scanning tp option, DeltaTau as step
const double Deltatp = 0.2;

double beta = 1;//0.2;//1;//5;
double eta  = 1; //0.2;//1;//5;
const double DAcoupling = 0.1;
double tp_fix = 5; //fixed t' for noneq FGR rate k(t',omega_DA) with scan omega_DA
const double DeltaTau =0.002; //time slice for t' griding
double Omega = 0.5; //primary mode freq
double y_0 = 1; //shift of primary mode
const double d_omega = omega_max / n_omega;//0.1;//0.002;for gaussian//0.1; for ohmic
const double d_omega_eff = omega_max / n_omega; //for effective SD sampling rate

const int LEN = 1024;//512;//1024; //number of t choices 1024 for gaussian//512 for ohmic
const double DeltaT=0.2;//0.2;//0.3; for gaussian//0.2 for ohmic //FFT time sampling interval
const double T0= -DeltaT*(LEN*0.5);//-DeltaT*LEN/2+DeltaT/2;

const double pi=3.14159265358979324;
const double RT_2PI= sqrt(2*pi);
const double hbar = 1;
//for gaussian spectral density
const double sigma = 0.1;
const double omega_op = 1.0;

void FFT(int dir, int m, double *x, double *y); //Fast Fourier Transform, 2^m data
double** Create_matrix(int row, int col);//new continuous 2D array in heap
double S_omega_ohmic(double omega, double eta); //ohmic with decay spectral density
double S_omega_drude(double omega, double eta);//another spectral density
double S_omega_gaussian(double omega, double eta, double sigma, double omega_op);//gaussian spectral density
double J_omega_ohmic(double omega, double eta);//bath Ohmic SD
double J_omega_ohmic_eff(double omega, double eta); //effective SD for Ohmic bath
void Integrand_LSC(double omega, double t, double &re, double &im);
void Integrand_NE_exact(double omega, double tp, double tau,  double shift, double req, double &re, double &im);
void Integrand_NE_CAV(double omega, double tp, double tau,  double shift, double req, double &re, double &im);
void Integrand_NE_CD(double omega, double tp, double tau,  double shift, double req, double &re, double &im);
void Integrand_NE_W0(double omega, double tp, double tau,  double shift, double req, double &re, double &im);
void Integrand_NE_Marcus(double omega, double tp, double tau,  double shift, double req, double &re, double &im);
void Linear_NE_exact(double omega, double tp, double tau,  double shift, double req, double &re, double &im);
void Linear_NE_LSC(double omega, double tp, double tau,  double shift, double req, double &re, double &im);
void Linear_NE_CAV(double omega, double tp, double tau,  double shift, double req, double &re, double &im);
void Linear_NE_CD(double omega, double tp, double tau,  double shift, double req, double &re, double &im);
void Linear_NE_W0(double omega, double tp, double tau,  double shift, double req, double &re, double &im);
void Linear_NE_Marcus(double omega, double tp, double tau,  double shift, double req, double &re, double &im);
double Integrate(double *data, int n, double dx);
double Sum(double *data, int n);


extern "C" {
    void dsyev_(const char &JOBZ, const char &UPLO, const int &N, double *A,
                const int &LDA, double *W, double *WORK, const int &LWORK,
                int &INFO);
}


int main (int argc, char *argv[]) {
    
    stringstream ss;
    string emptystr("");
    string filename;
    string idstr("");
    string nameapp;

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
    ofstream outfile1;
    
    double integral_re, integral_im;
    integral_re = integral_im = 0;
    
    double Er=0;
    double SD[n_omega];
    double J_eff[n_omega];
    
    int M; //time slice for tau = 0 --> tp
    int m; //index of tau
    double shift = T0 / DeltaT;
    double N = nn;
    //cout << "shift = " << shift << endl;
    double linear_accum_re;
    double linear_accum_im;
    double linear_re;
    double linear_im;
    double temp_re;
    double temp_im;
    double C_re;
    double C_im;
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
    
    double **TT_ns;
    TT_ns = Create_matrix(n_omega, n_omega);
    //transformation matrix: [normal mode]=[[TT_ns]]*[system-bath]
    //TT_ns * D_matrix * T_sn = diag, eigenvectors are row-vector of TT_ns
    double omega_nm[n_omega]; //normal mode frequencies
    double req_nm[n_omega]; //req of normal modes (acceptor shift)
    double c_nm[n_omega];//coupling strength of normal modes
    double S_array[n_omega];//Huang-Rhys factor for normal modes
    //double D_matrix[n_omega][n_omega];// the Hessian matrix
    double **D_matrix;
    D_matrix = Create_matrix(n_omega, n_omega);
    //double Diag_matrix[n_omega][n_omega]; //for testing diagonalization
    double gamma_nm[n_omega]; //linear coupling coefficient
    double shift_NE[n_omega]; //the s_j shifting for initial sampling
    
    double tp; //t' for noneq preparation
    tp = tp_fix;
    M = static_cast<int> (tp_fix/DeltaTau);
    double tau;
    double d_omega_DA = 2 * pi / LEN / DeltaT; //omega_DA griding size
    double omega_DA;
    double kre, kim;
    double sum(0);
    double kneq(0);
    
    int beta_index(0);
    int eta_index(0);
                  
    cout << "---- BEGIN of NEFGRL in non-Condon case ----" << endl;
    
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
    ss << "e" << eta << "_";
    nameapp = ss.str();

    
    //setting up spectral density for bath modes
    for (w = 0; w < n_omega; w++) J_eff[w] = J_omega_ohmic_eff(w*d_omega_eff, eta);
    for (w = 0; w < n_omega; w++) SD[w] = S_omega_ohmic(w*d_omega, eta); //Ohmic spectral density
    //for (w = 0; w < n_omega; w++) SD[w] = S_omega_drude(w*d_omega, eta); //Drude spectral density
    //for (w = 0; w < n_omega; w++) SD[w] = S_omega_gaussian(w*d_omega, eta, sigma, omega_op);
    
    for (w = 1; w < n_omega; w++) req_eff[w] = sqrt(8 * hbar * J_eff[w] / (pi * w * d_omega_eff*w * d_omega_eff*w));//eq min for each eff normal mode
    
    double c_bath[n_omega]; //secondary bath mode min shift coefficients
    for (w = 1; w < n_omega; w++) {
        c_bath[w] = sqrt( 2 / pi * J_omega_ohmic(w*d_omega, eta) * d_omega * d_omega * w);
    }
    
    // ********** BEGIN of Normal mode analysis ***********
    //construct Hessian matrix
    for (i=0; i< n_omega; i++) for (j=0; j<n_omega ;j++) D_matrix[i][j] = 0;
    D_matrix[0][0] = Omega*Omega;
    for (w =1 ; w < n_omega ; w++) {
        D_matrix[0][0] += pow(c_bath[w]/(w*d_omega) ,2);
        D_matrix[0][w] = D_matrix[w][0] = c_bath[w];
        D_matrix[w][w] = pow(w*d_omega ,2);
    }
    //copy Hessian matrix to working matrix for diagonalization
    for (i = 0; i < dim; i++) {
        for (j = 0; j < dim; j++) {
            matrix[j][i] = D_matrix[i][j];
            //NOTE: switch i j to match with Fortran array memory index
        }
    }
    
    //diagonalize matrix, the eigenvectors transpose is in result matrix => TT_ns.
    dsyev_('V', 'L', col, matrix[0], col, eig_val, work, lwork, info); //diagonalize matrix
    if (info != 0) cout << "Lapack failed. " << endl;
    
    for (i=0; i < dim; i++) omega_nm[i] = sqrt(eig_val[i]);
    
    for (i=0; i < dim; i++)
        for (j=0; j < dim; j++) TT_ns[i][j] = matrix[i][j];
    
    // the coefficients of linear electronic coupling in normal modes (gamma[j]=TT_ns[j][0]*gamma_y), here gamma_y=1
    
    //Noneq initial shift of each mode
    for (i=0; i<n_omega; i++) gamma_nm[i] = TT_ns[i][0];
    for (i=0; i<n_omega; i++) shift_NE[i] = s * gamma_nm[i];
    
    //req of normal modes (acceptor's potential energy min shift)
    for (i=0; i<n_omega; i++) {
        req_nm[i] = 1 * TT_ns[i][0];
        for (a=1; a < n_omega; a++) req_nm[i] -= TT_ns[i][a] * c_bath[a] / (a*d_omega * a*d_omega);
    }
    
    for (i=0; i<n_omega; i++) {
        //tilde c_j coupling strength normal mode
        c_nm[i] = req_nm[i] * omega_nm[i] * omega_nm[i];
        req_nm[i] *= 2 * y_0;
        //discrete Huang-Rhys factor
        S_array[i] = omega_nm[i] * req_nm[i] * req_nm[i] * 0.5;
    }
    outfile.close();
    outfile.clear();
    // ******** END of Normal mode analysis **************

    //we fix omega_DA, and scan tp = 0 - tp_max
    omega_DA = omega_DA_fix;
    
    ss.str("");
    idstr = "";
    ss << "s" << s;
    ss << "w" << omega_DA ;
    idstr += ss.str();
    
    
    //case [1]: exact for Noneq FGR in non-Condon case (linear coupling) using normal modes
    outfile.open((emptystr+"Exact_k_NEFGRL_"+nameapp+idstr+".dat").c_str());
    outfile1.open((emptystr+"Exact_P_NEFGRL_"+nameapp+idstr+".dat").c_str());
    sum=0;
    kneq=0;
    for (tp = 0; tp < tp_max; tp += Deltatp) {
        kre = 0;
        kim = 0;
        M = static_cast<int> (tp/DeltaTau);//update M for each tp
        for (m = 0; m < M; m++) {//tau index
            tau = m * DeltaTau;
            linear_accum_re = 0;
            linear_accum_im = 0;
            for (w = 0; w < n_omega; w++) {
                Integrand_NE_exact(omega_nm[w], tp, tau, shift_NE[w], req_nm[w], integ_re[w], integ_im[w]); //already multiplies S_array in Integrand_NE subroutine
                Linear_NE_exact(omega_nm[w], tp, tau, shift_NE[w], req_nm[w], linear_re, linear_im);
                linear_accum_re += linear_re * gamma_nm[w] * gamma_nm[w];
                linear_accum_im += linear_im * gamma_nm[w] * gamma_nm[w];
            }
            integral_re = Sum(integ_re, n_omega);
            integral_im = Sum(integ_im, n_omega);
            temp_re = exp(-1 * integral_re) * cos(integral_im);
            temp_im = exp(-1 * integral_re) * sin(-1 * integral_im);
            C_re = temp_re * linear_accum_re - temp_im * linear_accum_im;
            C_im = temp_re * linear_accum_im + temp_im * linear_accum_re;
            kre += C_re * cos(omega_DA*tau) - C_im * sin(omega_DA*tau);
            kim += C_re * sin(omega_DA*tau) + C_im * cos(omega_DA*tau);
        }
        kre *= DeltaTau;
        kim *= DeltaTau;
        kneq = kre*2*DAcoupling*DAcoupling;
        outfile << kneq << endl;
        sum += kneq * Deltatp;//probability of donor state
        //outfile1 << 1 - sum << endl; //1 - int dt' k(t')
        outfile1 << exp(-1*sum) << endl; //exp(- int dt' k(t'))
    }
    outfile.close();
    outfile.clear();
    outfile1.close();
    outfile1.clear();
    
    //case [2]: LSC for Noneq FGR in non-Condon case (linear coupling) using normal modes
    outfile.open((emptystr+"LSC_k_NEFGRL_"+nameapp+idstr+".dat").c_str());
    outfile1.open((emptystr+"LSC_P_NEFGRL_"+nameapp+idstr+".dat").c_str());
    sum=0;
    kneq=0;
    for (tp = 0; tp < tp_max; tp += Deltatp) {
        kre = 0;
        kim = 0;
        M = static_cast<int> (tp/DeltaTau);//update M for each tp
        for (m = 0; m < M; m++) {//tau index
            tau = m * DeltaTau;
            linear_accum_re = 0;
            linear_accum_im = 0;
            for (w = 0; w < n_omega; w++) {
                Integrand_NE_exact(omega_nm[w], tp, tau, shift_NE[w], req_nm[w], integ_re[w], integ_im[w]); //already multiplies S_array in Integrand_NE subroutine
                Linear_NE_LSC(omega_nm[w], tp, tau, shift_NE[w], req_nm[w], linear_re, linear_im);
                linear_accum_re += linear_re * gamma_nm[w] * gamma_nm[w];
                linear_accum_im += linear_im * gamma_nm[w] * gamma_nm[w];
            }
            integral_re = Sum(integ_re, n_omega);
            integral_im = Sum(integ_im, n_omega);
            temp_re = exp(-1 * integral_re) * cos(integral_im);
            temp_im = exp(-1 * integral_re) * sin(-1 * integral_im);
            C_re = temp_re * linear_accum_re - temp_im * linear_accum_im;
            C_im = temp_re * linear_accum_im + temp_im * linear_accum_re;
            kre += C_re * cos(omega_DA*tau) - C_im * sin(omega_DA*tau);
            kim += C_re * sin(omega_DA*tau) + C_im * cos(omega_DA*tau);
        }
        kre *= DeltaTau;
        kim *= DeltaTau;
        kneq = kre*2*DAcoupling*DAcoupling;
        outfile << kneq << endl;
        sum += kneq * Deltatp;//probability of donor state
        //outfile1 << 1 - sum << endl; //1 - int dt' k(t')
        outfile1 << exp(-1*sum) << endl; //exp(- int dt' k(t'))
    }
    outfile.close();
    outfile.clear();
    outfile1.close();
    outfile1.clear();

    //case [3]: CAV for Noneq FGR in non-Condon case (linear coupling) using normal modes
    outfile.open((emptystr+"CAV_k_NEFGRL_"+nameapp+idstr+".dat").c_str());
    outfile1.open((emptystr+"CAV_P_NEFGRL_"+nameapp+idstr+".dat").c_str());
    sum=0;
    kneq=0;
    for (tp = 0; tp < tp_max; tp += Deltatp) {
        kre = 0;
        kim = 0;
        M = static_cast<int> (tp/DeltaTau);//update M for each tp
        for (m = 0; m < M; m++) {//tau index
            tau = m * DeltaTau;
            linear_accum_re = 0;
            linear_accum_im = 0;
            for (w = 0; w < n_omega; w++) {
                Integrand_NE_CAV(omega_nm[w], tp, tau, shift_NE[w], req_nm[w], integ_re[w], integ_im[w]); //already multiplies S_array in Integrand_NE subroutine
                Linear_NE_CAV(omega_nm[w], tp, tau, shift_NE[w], req_nm[w], linear_re, linear_im);
                linear_accum_re += linear_re * gamma_nm[w] * gamma_nm[w];
                linear_accum_im += linear_im * gamma_nm[w] * gamma_nm[w];
            }
            integral_re = Sum(integ_re, n_omega);
            integral_im = Sum(integ_im, n_omega);
            temp_re = exp(-1 * integral_re) * cos(integral_im);
            temp_im = exp(-1 * integral_re) * sin(-1 * integral_im);
            C_re = temp_re * linear_accum_re - temp_im * linear_accum_im;
            C_im = temp_re * linear_accum_im + temp_im * linear_accum_re;
            kre += C_re * cos(omega_DA*tau) - C_im * sin(omega_DA*tau);
            kim += C_re * sin(omega_DA*tau) + C_im * cos(omega_DA*tau);
        }
        kre *= DeltaTau;
        kim *= DeltaTau;
        kneq = kre*2*DAcoupling*DAcoupling;
        outfile << kneq << endl;
        sum += kneq * Deltatp;//probability of donor state
        //outfile1 << 1 - sum << endl; //1 - int dt' k(t')
        outfile1 << exp(-1*sum) << endl; //exp(- int dt' k(t'))
    }
    outfile.close();
    outfile.clear();
    outfile1.close();
    outfile1.clear();
    
    
    //case [4]: CD for Noneq FGR in non-Condon case (linear coupling) using normal modes
    outfile.open((emptystr+"CD_k_NEFGRL_"+nameapp+idstr+".dat").c_str());
    outfile1.open((emptystr+"CD_P_NEFGRL_"+nameapp+idstr+".dat").c_str());
    sum=0;
    kneq=0;
    for (tp = 0; tp < tp_max; tp += Deltatp) {
        kre = 0;
        kim = 0;
        M = static_cast<int> (tp/DeltaTau);//update M for each tp
        for (m = 0; m < M; m++) {//tau index
            tau = m * DeltaTau;
            linear_accum_re = 0;
            linear_accum_im = 0;
            for (w = 0; w < n_omega; w++) {
                Integrand_NE_CD(omega_nm[w], tp, tau, shift_NE[w], req_nm[w], integ_re[w], integ_im[w]); //already multiplies S_array in Integrand_NE subroutine
                Linear_NE_CD(omega_nm[w], tp, tau, shift_NE[w], req_nm[w], linear_re, linear_im);
                linear_accum_re += linear_re * gamma_nm[w] * gamma_nm[w];
                linear_accum_im += linear_im * gamma_nm[w] * gamma_nm[w];
            }
            integral_re = Sum(integ_re, n_omega);
            integral_im = Sum(integ_im, n_omega);
            temp_re = exp(-1 * integral_re) * cos(integral_im);
            temp_im = exp(-1 * integral_re) * sin(-1 * integral_im);
            C_re = temp_re * linear_accum_re - temp_im * linear_accum_im;
            C_im = temp_re * linear_accum_im + temp_im * linear_accum_re;
            kre += C_re * cos(omega_DA*tau) - C_im * sin(omega_DA*tau);
            kim += C_re * sin(omega_DA*tau) + C_im * cos(omega_DA*tau);
        }
        kre *= DeltaTau;
        kim *= DeltaTau;
        kneq = kre*2*DAcoupling*DAcoupling;
        outfile << kneq << endl;
        sum += kneq * Deltatp;//probability of donor state
        //outfile1 << 1 - sum << endl; //1 - int dt' k(t')
        outfile1 << exp(-1*sum) << endl; //exp(- int dt' k(t'))
    }
    outfile.close();
    outfile.clear();
    outfile1.close();
    outfile1.clear();
    
    //case [5]: W0 for Noneq FGR in non-Condon case (linear coupling) using normal modes
    outfile.open((emptystr+"inh_k_NEFGRL_"+nameapp+idstr+".dat").c_str());
    outfile1.open((emptystr+"inh_P_NEFGRL_"+nameapp+idstr+".dat").c_str());
    sum=0;
    kneq=0;
    for (tp = 0; tp < tp_max; tp += Deltatp) {
        kre = 0;
        kim = 0;
        M = static_cast<int> (tp/DeltaTau);//update M for each tp
        for (m = 0; m < M; m++) {//tau index
            tau = m * DeltaTau;
            linear_accum_re = 0;
            linear_accum_im = 0;
            for (w = 0; w < n_omega; w++) {
                Integrand_NE_W0(omega_nm[w], tp, tau, shift_NE[w], req_nm[w], integ_re[w], integ_im[w]); //already multiplies S_array in Integrand_NE subroutine
                Linear_NE_W0(omega_nm[w], tp, tau, shift_NE[w], req_nm[w], linear_re, linear_im);
                linear_accum_re += linear_re * gamma_nm[w] * gamma_nm[w];
                linear_accum_im += linear_im * gamma_nm[w] * gamma_nm[w];
            }
            integral_re = Sum(integ_re, n_omega);
            integral_im = Sum(integ_im, n_omega);
            temp_re = exp(-1 * integral_re) * cos(integral_im);
            temp_im = exp(-1 * integral_re) * sin(-1 * integral_im);
            C_re = temp_re * linear_accum_re - temp_im * linear_accum_im;
            C_im = temp_re * linear_accum_im + temp_im * linear_accum_re;
            kre += C_re * cos(omega_DA*tau) - C_im * sin(omega_DA*tau);
            kim += C_re * sin(omega_DA*tau) + C_im * cos(omega_DA*tau);
        }
        kre *= DeltaTau;
        kim *= DeltaTau;
        kneq = kre*2*DAcoupling*DAcoupling;
        outfile << kneq << endl;
        sum += kneq * Deltatp;//probability of donor state
        //outfile1 << 1 - sum << endl; //1 - int dt' k(t')
        outfile1 << exp(-1*sum) << endl; //exp(- int dt' k(t'))
    }
    outfile.close();
    outfile.clear();
    outfile1.close();
    outfile1.clear();
    
    //case [6]: W0 for Noneq FGR in non-Condon case (linear coupling) using normal modes
    outfile.open((emptystr+"Marcus_k_NEFGRL_"+nameapp+idstr+".dat").c_str());
    outfile1.open((emptystr+"Marcus_P_NEFGRL_"+nameapp+idstr+".dat").c_str());
    sum=0;
    kneq=0;
    for (tp = 0; tp < tp_max; tp += Deltatp) {
        kre = 0;
        kim = 0;
        M = static_cast<int> (tp/DeltaTau);//update M for each tp
        for (m = 0; m < M; m++) {//tau index
            tau = m * DeltaTau;
            linear_accum_re = 0;
            linear_accum_im = 0;
            for (w = 0; w < n_omega; w++) {
                Integrand_NE_Marcus(omega_nm[w], tp, tau, shift_NE[w], req_nm[w], integ_re[w], integ_im[w]); //already multiplies S_array in Integrand_NE subroutine
                Linear_NE_Marcus(omega_nm[w], tp, tau, shift_NE[w], req_nm[w], linear_re, linear_im);
                linear_accum_re += linear_re * gamma_nm[w] * gamma_nm[w];
                linear_accum_im += linear_im * gamma_nm[w] * gamma_nm[w];
            }
            integral_re = Sum(integ_re, n_omega);
            integral_im = Sum(integ_im, n_omega);
            temp_re = exp(-1 * integral_re) * cos(integral_im);
            temp_im = exp(-1 * integral_re) * sin(-1 * integral_im);
            C_re = temp_re * linear_accum_re - temp_im * linear_accum_im;
            C_im = temp_re * linear_accum_im + temp_im * linear_accum_re;
            kre += C_re * cos(omega_DA*tau) - C_im * sin(omega_DA*tau);
            kim += C_re * sin(omega_DA*tau) + C_im * cos(omega_DA*tau);
        }
        kre *= DeltaTau;
        kim *= DeltaTau;
        kneq = kre*2*DAcoupling*DAcoupling;
        outfile << kneq << endl;
        sum += kneq * Deltatp;//probability of donor state
        //outfile1 << 1 - sum << endl; //1 - int dt' k(t')
        outfile1 << exp(-1*sum) << endl; //exp(- int dt' k(t'))
    }
    outfile.close();
    outfile.clear();
    outfile1.close();
    outfile1.clear();
    
    
    case_count++;

    //-------------- Summary ----------------
    
    cout << "CASE # " << case_count <<  " done:" << endl;
    cout << "   beta = " << beta << endl;
    cout << "    eta = " << eta << endl;
    cout << "-------------------" << endl;
    
    
    }
    
    
    
    cout << "--- SUMMARY --- " << endl;
    cout << "   fix omega_DA = " << omega_DA_fix << endl;
    cout << "   Delta tp = " << Deltatp << endl;
    cout << "   number of tp = " << tp_max/Deltatp << endl << endl;
    cout << "   normal modes n_omega = " << n_omega << endl;
    cout << "   initial shift s = " << s << endl;
    cout << "---- END of all NEFGRL in non-Condon case ----" << endl;
    return 0;
}



/********* SUBROUTINE *************/

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
    X(n) = ----  >   x(k) e                   = forward transform
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


//spectral densities

double S_omega_ohmic(double omega, double eta) {
    return eta * omega * exp(-1 * omega);
}

double S_omega_drude(double omega, double eta) {
    return eta * omega /(1 + omega*omega);
}

double S_omega_gaussian(double omega, double eta, double sigma, double omega_op) {
    return   0.5 / hbar * eta * omega * exp(-(omega - omega_op)*(omega - omega_op)/(2*sigma*sigma))/RT_2PI/sigma;
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

//min-to-min energy as Fourier transform frequency
void Integrand_LSC(double omega, double t, double &re, double &im) {
    re = (1-cos(omega*t))/tanh(beta*hbar*omega/2);
    im = sin(omega*t);
    return;
}

void Integrand_NE_exact(double omega, double tp, double tau, double shift, double req, double &re, double &im) {//including Huang-Rhys factor S_j
    re = omega*req*req*0.5*(1-cos(omega*tau))/tanh(beta*hbar*omega/2);
    im = omega*req*req*0.5*sin(omega*tau) + omega*req*shift* (sin(omega*tp) + sin(omega*tau - omega*tp));
    return;
}


void Integrand_NE_CAV(double omega, double tp, double tau, double shift, double req, double &re, double &im) {
    re = req*req/beta*(1-cos(omega*tau));
    im = omega*req*req*0.5*sin(omega*tau) + omega*req*shift * (sin(omega*tp) + sin(omega*tau - omega*tp));
    return;
}

void Integrand_NE_CD(double omega, double tp, double tau, double shift, double req, double &re, double &im) {
    re = req*req/beta*(1-cos(omega*tau));
    im = omega*req*req*0.5 * omega*tau + omega*req*shift * (sin(omega*tp) + sin(omega*tau - omega*tp));
    return;
}

void Integrand_NE_W0(double omega, double tp, double tau, double shift, double req, double &re, double &im) {
    re = omega*req*req*0.5 / tanh(beta*hbar*omega/2) * omega*omega*tau*tau*0.5;
    im = omega*req*req*0.5 * omega*tau + omega*req*shift * cos(omega*tp) * omega*tau;
    return;
}

void Integrand_NE_Marcus(double omega, double tp, double tau, double shift, double req, double &re, double &im) {
    re = omega*req*req*0.5 * omega*tau*tau / beta;
    im = omega*req*req*0.5 * omega*tau + omega*req*shift * cos(omega*tp) * omega*tau;
    return;
}


void Linear_NE_exact(double omega, double tp, double tau,  double shift, double req, double &re, double &im) {
    double Coth = 1.0 / tanh(beta*hbar*omega*0.5);
    double u1_re(0), u1_im(0), u2_re(0), u2_im(0);
    u1_re = 0.5 * hbar / omega * Coth * cos(omega * tau);
    u1_im = -0.5 * hbar / omega * sin(omega * tau);
    u2_re = req * (1-cos(omega * tau));
    u2_im = req * Coth * sin(omega * tau);
    re = u1_re + 0.25 * (u2_re - 2*shift*cos(omega * tp)) * (u2_re - 2*shift*cos(omega * tp - omega * tau)) - 0.25 * u2_im * u2_im;
    im = u1_im + 0.25 * (u2_re - 2*shift*cos(omega * tp)) * u2_im + 0.25 * (u2_re - 2*shift*cos(omega * tp - omega * tau)) * u2_im;
    return;
}


void Linear_NE_LSC(double omega, double tp, double tau,  double shift, double req, double &re, double &im) {
    double Coth = 1.0 / tanh(beta*hbar*omega*0.5);
    re = shift*shift*cos(omega*tp)*cos(omega*tp-omega*tau) + Coth * hbar/omega*0.5*cos(omega*tau) - req*req* 0.5 * Coth * Coth * pow(sin(0.5*omega*tau),2) * (cos(4*omega*tp - 2*omega*tau) + cos(omega*tau)) ;
    im =  0.25*req*shift * Coth * ( (1-2*cos(omega*tau))*sin(omega*tp) + sin(omega*tp-2*omega*tau) - 4* cos(3*omega*tp - 1.5*omega*tau)*sin(0.5*omega*tau) );
    return;
}


void Linear_NE_CAV(double omega, double tp, double tau,  double shift, double req, double &re, double &im) {
    re = shift*shift*cos(omega*tp)*cos(omega*tp-omega*tau) + 1.0/(beta*omega*omega) * cos(omega*tau) - req*req* 0.5 / pow(beta*hbar*omega*0.5,2) * pow(sin(0.5*omega*tau),2) * (cos(4*omega*tp - 2*omega*tau) + cos(omega*tau)) ;
    im =  0.5*req*shift / (beta*hbar*omega) * ( (1-2*cos(omega*tau))*sin(omega*tp) + sin(omega*tp-2*omega*tau) - 4* cos(3*omega*tp - 1.5*omega*tau)*sin(0.5*omega*tau) );
    return;
}

void Linear_NE_CD(double omega, double tp, double tau,  double shift, double req, double &re, double &im) {
    re = shift*shift*cos(omega*tp)*cos(omega*tp-omega*tau) + cos(omega*tau)/(beta*omega*omega) -  pow(req*sin(omega*tau)/(beta*hbar*omega),2);
    im = -4.0*shift*req/(beta*hbar*omega) * cos(omega*tp-0.5*omega*tau) * pow(cos(0.5*omega*tau),2) * sin(0.5*omega*tau);
    return;
}

void Linear_NE_W0(double omega, double tp, double tau,  double shift, double req, double &re, double &im) {
    double Coth = 1.0 / tanh(beta*hbar*omega*0.5);
    re = Coth*hbar*0.5/omega + 0.5*shift*shift*(1+cos(2*omega*tp)) - pow(req*omega*tau*0.5, 2)*Coth*Coth;
    im = - Coth * req * shift * omega * tau * cos(omega*tp);
    return;
}

void Linear_NE_Marcus(double omega, double tp, double tau,  double shift, double req, double &re, double &im) {
    re = 1.0 / (beta*omega*omega) + 0.5*shift*shift*(1+cos(2*omega*tp)) - pow(req*tau/beta/hbar, 2);
    im = - 2.0 * req * shift * tau / (beta*hbar) * cos(omega*tp);
    return;
}


double Integrate(double *data, int n, double dx){
    double I =0;
    I += (data[0]+data[n-1])/2;
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







