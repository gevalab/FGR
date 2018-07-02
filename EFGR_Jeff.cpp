/* This code calculates equilibrium Fermi Golden Rule rate
 in Condon case, Brownian oscillator model
 compare with linearized semiclassical methods
 [special case with Jeff spectral density for ohmic bath modes]
 To compile: g++ -o EFGR_Jeff EFGR_Jeff.cpp -llapack -lrefblas -lgfortran
 (c) Xiang Sun 2015
 */

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
using namespace std;

// *********** change parameters *********
const int bldim = 1;//3;
const int eldim = 1;// 3;
double beta_list[bldim] = {1};// {0.2, 1.0, 5.0};  //{0.1, 1, 10}; //{0.2, 1.0, 5.0};
double eta_list[eldim] = {1}; //{0.5, 1.0, 5.0}; //{0.1, 1, 10}; //{0.5, 1.0, 5.0};
double Omega = 2;//0.5; //primary mode freq 0.2, 0.5, 1, 2
double y_0 = 1;//10;//sqrt(10.0);//1.0; //shift of primary mode
const double omega_max = 15;//40;//20;//15 or 20 for Jeff
const int n_omega = 10000;//25000;//10000;
const int LEN = 1024;//1024;//512; //number of t choices
const double DeltaT = 0.05;//0.2; //FFT time sampling interval
// *********** **************** *********


double beta = 1;//0.2, 0.5;//1;//5;
double eta = 1; //0.5;//1;//5;
const double DAcoupling = 0.1;
const double d_omega = omega_max / n_omega;//0.1;
const double d_omega_eff = omega_max / n_omega;//0.05; //for effective SD sampling rate
const double omega_c = 1; //cutoff freq for ohmic
const double T0= -DeltaT*(LEN*0.5);
const double hbar = 1;
const double pi=3.14159265358979324;
const double RT_2PI= sqrt(2*pi);

//Declare Subroutines
void FFT(int dir, int m, double *x, double *y); //Fast Fourier Transform, 2^m data
void DFT(int dir, int m, double *x, double *y); //Discrete Fourier Transform
double S_omega_ohmic(double omega, double eta); //ohmic with decay spectral density
double S_omega_drude(double omega, double eta);//another spectral density
double S_omega_gaussian(double omega, double eta, double sigma, double omega_op);//gaussian spectral density
double J_omega_ohmic(double omega, double eta);//bath Ohmic SD
double J_omega_ohmic_eff(double omega, double eta); //effective SD for Ohmic bath
void Integrand_LSC(double omega, double t, double &re, double &im);
void Integrand_LSC_inh(double omega, double t, double &re, double &im);
void Integrand_CL_avg(double omega, double t, double &re, double &im);
void Integrand_CL_donor(double omega, double t, double &re, double &im);
void Integrand_2cumu(double omega, double t, double &re, double &im);
void Integrand_2cumu_inh(double omega, double t, double &re, double &im);
void Integrand_NE_exact(double omega, double tp, double tau,  double shift, double req, double &re, double &im);
double Integrate(double *data, int n, double dx);
double Integrate_from(double *data, int sp, int n, double dx);
double Sum(double *data, int n);
double** Create_matrix(int row, int col);//new continuous 2D array in heap

extern "C" {
    void dsyev_(const char &JOBZ, const char &UPLO, const int &N, double *A,
                const int &LDA, double *W, double *WORK, const int &LWORK,
                int &INFO);
}


int main (int argc, char *argv[]) {
    
    stringstream ss;
    string emptystr("");
    string filename("");
    string idstr("");
    string nameapp("");
    
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
    ofstream outfile2;
    
    double integral_re, integral_im;
    integral_re = integral_im = 0;
    
    double J_eff[n_omega]; // Jeff  for Omega << omega_c
    double J_eff2[n_omega];// Jeff2 for Omega >> omega_c
    
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
    double c_eff[n_omega];//c from "effective" SD
    double req_eff[n_omega];//req from "effective" SD
    double Er(0); //reorganization energy from discrete normal modes
    double Er_bath(0);//reorganization energy for Ohmic bath
    double Er_eff(0); //reorganization energy for effective drude SD
    double Er_eff_Jw(0);
    double Er_eff_RRww(0);
    double a_parameter(0);
    double a_parameter_eff(0);
    
    double tau_c(0);
    int shift_index = static_cast<int>(-shift);
    
    int beta_index(0);
    int eta_index(0);
    
    cout << "---------- Eq FGR in Condon case ----------" << endl;
    outfile2.open((emptystr + "Tau_c_EFGR_Jeff.dat").c_str());
    // Part I - Using exact discrete normal modes
    
    //BEGIN loop through thermal conditions
    int case_count(0);
    for (beta_index = 0; beta_index < bldim; beta_index++)
    for (eta_index = 0; eta_index < eldim; eta_index++)
    {
        beta = beta_list[beta_index];
        eta = eta_list[eta_index];
        ss.str("");
        nameapp = "";
        ss << "Omega"<< Omega << "_";
        ss << "b" << beta;
        ss << "e" << eta;
        nameapp = ss.str();
            
            
    //setting up spectral density
    for (w = 1; w < n_omega; w++) J_eff[w] = J_omega_ohmic_eff(w*d_omega_eff, eta);//Jeff1
    for (w = 1; w < n_omega; w++) J_eff2[w] = J_omega_ohmic(w*d_omega_eff, eta); //Jeff2
    
    //outfile1.open("J_eff(omega).dat");
    //for (w = 1; w< n_omega; w++) outfile1 << J_eff[w] << endl;
    //outfile1.close();
    //outfile1.clear();
    
    Er_eff_RRww = 0;
    Er_eff_Jw = 0;
    a_parameter_eff = 0;
    
    for (w = 1; w < n_omega; w++) {
        //eq min for each "eff" normal mode
        req_eff[w] = sqrt(8.0 * d_omega_eff * J_eff[w] / pi / pow(w*d_omega_eff,3));
        //c_alpha for each "eff" normal mode
        c_eff[w] = sqrt( 2.0 / pi * J_eff[w] * d_omega_eff * d_omega_eff * w);
        
        Er_eff_Jw += J_eff[w] / (w * d_omega_eff) * d_omega_eff * 4.0/pi;
        Er_eff_RRww += 0.5 * req_eff[w] * req_eff[w]  * w * d_omega_eff * w * d_omega_eff;
        
        a_parameter_eff += 0.25 * pow(w * d_omega_eff,3) *req_eff[w]*req_eff[w] /tanh(beta*hbar* w * d_omega_eff *0.5);
    }
    //cout << "Er_eff = " << Er_eff << endl;
    
    //cout << "Er_eff_Jw   = " << Er_eff_Jw << endl;
    //cout << "Er_eff_RRww = " << Er_eff_RRww << endl;
    //outfile << Er_eff_Jw  << endl;
    
    Er_eff = Er_eff_Jw;
    
    //cout << "a_parameter_eff = " << a_parameter_eff << endl;

    
    
    //=============case: [1] Eq FGR using continuous SD J_eff(\omega)=============
    
    //[a] Exact or LSC approximation using Jeff
    outfile1.open((emptystr + "EFGR_Integral_Jeff_" + nameapp + ".dat").c_str());
    //outfile1.open((emptystr + "QMLSC_EFGR_Jeff_tre_" + nameapp + ".dat").c_str());
    
    for (i = 0; i < nn; i++) corr1[i] = corr2[i] = 0; //zero padding
    for (i = 0; i < LEN; i++) {
        t = T0 + DeltaT * i;
        integ_re[0] =0;
        integ_im[0] =0;
        for (w = 1; w < n_omega; w++) {
            omega = w * d_omega_eff;
            //omega = (w+1) * d_omega_eff;
            Integrand_LSC(omega, t, integ_re[w], integ_im[w]);
            integ_re[w] *= 4*J_eff[w]/pi/(omega*omega);
            integ_im[w] *= 4*J_eff[w]/pi/(omega*omega);
        }
        integral_re = Integrate_from(integ_re, 1, n_omega, d_omega_eff);
        integral_im = Integrate_from(integ_im, 1, n_omega, d_omega_eff);
        //integral_re = Integrate(integ_re, n_omega, d_omega_eff);
        //integral_im = Integrate(integ_im, n_omega, d_omega_eff);
        
        outfile1 << integral_re << "\t" << integral_im << endl;
        
        corr1[i] = exp(-1 * integral_re) * cos(integral_im);
        corr2[i] = -1 * exp(-1 * integral_re) * sin(integral_im);
        
        //outfile1 << corr1[i] << endl;
    }
    
    tau_c = 0.5 * Integrate(corr1, nn, DeltaT) / corr1[shift_index];
    outfile2 << tau_c << endl;
        
    FFT(-1, mm, corr1, corr2);//notice its inverse FT
    
    for(i=0; i<nn; i++) { //shift time origin
        corr1_orig[i] = corr1[i] * cos(2*pi*i*shift/N) - corr2[i] * sin(-2*pi*i*shift/N);
        corr2_orig[i] = corr2[i] * cos(2*pi*i*shift/N) + corr1[i] * sin(-2*pi*i*shift/N);
    }
    
    outfile.open((emptystr + "QMLSC_EFGR_Jeff_" + nameapp + ".dat").c_str());
    for (i=0; i<nn/2; i++) outfile << corr1_orig[i]*LEN*DeltaT*DAcoupling*DAcoupling << endl;
    outfile.close();
    outfile.clear();
    outfile1.close();
    outfile1.clear();
    
    
    //[a'] Exact or LSC approximation using Jeff2
    outfile1.open((emptystr + "EFGR_Integral_Jeff2_" + nameapp + ".dat").c_str());
    for (i = 0; i < nn; i++) corr1[i] = corr2[i] = 0; //zero padding
    for (i = 0; i < LEN; i++) {
        t = T0 + DeltaT * i;
        integ_re[0] =0;
        integ_im[0] =0;
        for (w = 1; w < n_omega; w++) {
            omega = w * d_omega_eff;
            //omega = (w+1) * d_omega_eff;
            Integrand_LSC(omega, t, integ_re[w], integ_im[w]);
            integ_re[w] *= 4*J_eff2[w]/pi/(omega*omega);
            integ_im[w] *= 4*J_eff2[w]/pi/(omega*omega);
        }
        Integrand_LSC(Omega, t, temp_re, temp_im);
        integral_re = Integrate_from(integ_re, 1, n_omega, d_omega_eff);
        integral_re += 2 * Omega * y_0 * y_0 * temp_re;//the Primary mode contribution
        integral_im = Integrate_from(integ_im, 1, n_omega, d_omega_eff);
        integral_im += 2 * Omega * y_0 * y_0 * temp_im;//the Primary mode contribution
        
        outfile1 << integral_re << "\t" << integral_im << endl;
        corr1[i] = exp(-1 * integral_re) * cos(integral_im);
        corr2[i] = -1 * exp(-1 * integral_re) * sin(integral_im);
    }
    
    FFT(-1, mm, corr1, corr2);//notice its inverse FT
    
    for(i=0; i<nn; i++) { //shift time origin
        corr1_orig[i] = corr1[i] * cos(2*pi*i*shift/N) - corr2[i] * sin(-2*pi*i*shift/N);
        corr2_orig[i] = corr2[i] * cos(2*pi*i*shift/N) + corr1[i] * sin(-2*pi*i*shift/N);
    }
    
    outfile.open((emptystr + "QMLSC_EFGR_Jeff2_" + nameapp + ".dat").c_str());
    for (i=0; i<nn/2; i++) outfile << corr1_orig[i]*LEN*DeltaT*DAcoupling*DAcoupling << endl;
    outfile.close();
    outfile.clear();
    outfile1.close();
    outfile1.clear();
    
    
    //[b] inh approximation
    for (i = 0; i < nn; i++) corr1[i] = corr2[i] = 0; //zero padding
    for (i = 0; i < LEN; i++) {
        t = T0 + DeltaT * i;
        integ_re[0] =0;
        integ_im[0] =0;
        for (w = 1; w < n_omega; w++) {
            omega = w * d_omega_eff;
            Integrand_LSC_inh(omega, t, integ_re[w], integ_im[w]);
            integ_re[w] *= 4*J_eff[w]/pi/(omega*omega);
            integ_im[w] *= 4*J_eff[w]/pi/(omega*omega);
        }
        integral_re = Integrate_from(integ_re, 1, n_omega, d_omega_eff);
        integral_im = Integrate_from(integ_im, 1, n_omega, d_omega_eff);
        
        corr1[i] = exp(-1 * integral_re) * cos(integral_im);
        corr2[i] = -1 * exp(-1 * integral_re) * sin(integral_im);
    }
    
    FFT(-1, mm, corr1, corr2);//notice its inverse FT
    
    for(i=0; i<nn; i++) { //shift time origin
        corr1_orig[i] = corr1[i] * cos(2*pi*i*shift/N) - corr2[i] * sin(-2*pi*i*shift/N);
        corr2_orig[i] = corr2[i] * cos(2*pi*i*shift/N) + corr1[i] * sin(-2*pi*i*shift/N);
    }
    
    outfile.open((emptystr + "W0_EFGR_Jeff_" + nameapp + ".dat").c_str());
    //outfile.open("inh_EFGR_Jeff.dat");
    for (i=0; i<nn/2; i++) outfile << corr1_orig[i]*LEN*DeltaT*DAcoupling*DAcoupling << endl;
    outfile.close();
    outfile.clear();
    
    
    //[c] C-AV approximation: classical sampling with dynamics on average surface
    for (i = 0; i < nn; i++) corr1[i] = corr2[i] = 0; //zero padding
    for (i = 0; i < LEN; i++) {
        t = T0 + DeltaT * i;
        integ_re[0] = 0;
        integ_im[0] = 0;
        for (w = 1; w < n_omega; w++) {
            omega = w * d_omega_eff;
            Integrand_CL_avg(omega, t, integ_re[w], integ_im[w]);
            integ_re[w] *= 4*J_eff[w]/pi/(omega*omega);
            integ_im[w] *= 4*J_eff[w]/pi/(omega*omega);
        }
        integral_re = Integrate_from(integ_re, 1, n_omega, d_omega_eff);
        integral_im = Integrate_from(integ_im, 1, n_omega, d_omega_eff);
        
        corr1[i] = exp(-1 * integral_re) * cos(integral_im);
        corr2[i] = -1 * exp(-1 * integral_re) * sin(integral_im);
    }
    
    FFT(-1, mm, corr1, corr2);//notice its inverse FT
    
    for(i=0; i<nn; i++) { //shift time origin
        corr1_orig[i] = corr1[i] * cos(2*pi*i*shift/N) - corr2[i] * sin(-2*pi*i*shift/N);
        corr2_orig[i] = corr2[i] * cos(2*pi*i*shift/N) + corr1[i] * sin(-2*pi*i*shift/N);
    }
    
    outfile.open((emptystr + "CAV_EFGR_Jeff_" + nameapp + ".dat").c_str());
    for (i=0; i<nn/2; i++) outfile << corr1_orig[i]*LEN*DeltaT*DAcoupling*DAcoupling << endl;
    outfile.close();
    outfile.clear();
    
    
    //[d] C-D approximation: Classical sampling with donor potential dynamics (freq shifting)
    for (i = 0; i < nn; i++) corr1[i] = corr2[i] = 0;
    for (i=0; i< LEN; i++) {
        t = T0 + DeltaT * i;
        integ_re[0]=integ_im[0]=0;
        for (w = 1; w < n_omega; w++) {
            omega = w * d_omega_eff;
            Integrand_CL_donor(omega, t, integ_re[w], integ_im[w]);
            integ_re[w] *= 4*J_eff[w]/pi/(omega*omega);
            integ_im[w] *= 4*J_eff[w]/pi/(omega*omega);
        }
        integral_re = Integrate_from(integ_re, 1, n_omega, d_omega_eff);
        //integral_im = Integrate_from(integ_im, 1, n_omega, d_omega_eff);
        
        corr1[i] = exp(-1 * integral_re) ;
        corr2[i] = 0;
        //corr1[i] = exp(-1 * integral_re) * cos(integral_im);
        //corr2[i] = -1 * exp(-1 * integral_re) * sin(integral_im);
    }
    
    FFT(-1, mm, corr1, corr2);
    
    for(i=0; i<nn; i++) { //shift time origin
        corr1_orig[i] = corr1[i] * cos(2*pi*i*shift/N) - corr2[i] * sin(-2*pi*i*shift/N);
        corr2_orig[i] = corr2[i] * cos(2*pi*i*shift/N) + corr1[i] * sin(-2*pi*i*shift/N);
    }
    
    //shift freq -omega_0=-Er_eff
    int shift_f(0);
    shift_f = static_cast<int> (Er_eff/(1.0/LEN/DeltaT)/(2*pi)+0.5);
    
    outfile.open((emptystr + "CD_EFGR_Jeff_" + nameapp + ".dat").c_str());
    for (i=nn-shift_f; i<nn; i++) outfile << corr1_orig[i]*LEN*DeltaT*DAcoupling*DAcoupling << endl;
    for (i=0; i<nn-shift_f; i++) outfile << corr1_orig[i]*LEN*DeltaT*DAcoupling*DAcoupling << endl;
    outfile.close();
    outfile.clear();
    
    
    //[e] Marcus approximation: second order cumulant + inhomogeneous limit (Marcus)
    /*
    for (i = 0; i < nn; i++) corr1[i] = corr2[i] = 0;
    for (i=0; i< LEN; i++) {
        t = T0 + DeltaT * i;
        integ_re[0]=integ_im[0]=0;
        for (w = 1; w < n_omega; w++) {
            omega = w * d_omega_eff;
            Integrand_2cumu_inh(omega, t, integ_re[w], integ_im[w]);
            integ_re[w] *= 4*J_eff[w]/pi/(omega*omega);
            integ_im[w] *= 4*J_eff[w]/pi/(omega*omega);
        }
        integral_re = Integrate_from(integ_re, 1, n_omega, d_omega_eff);
        integral_im = Integrate_from(integ_im, 1, n_omega, d_omega_eff);
        
        corr1[i] = exp(-1 * integral_re) * cos(integral_im);
        corr2[i] = -1 * exp(-1 * integral_re) * sin(integral_im);
    }
    
    FFT(-1, mm, corr1, corr2);
    
    for(i=0; i<nn; i++) { //shift time origin
        corr1_orig[i] = corr1[i] * cos(2*pi*i*shift/N) - corr2[i] * sin(-2*pi*i*shift/N);
        corr2_orig[i] = corr2[i] * cos(2*pi*i*shift/N) + corr1[i] * sin(-2*pi*i*shift/N);
    }
    
    outfile.open("marcus_EFGR_Jeff.dat");
    for (i=0; i<nn/2; i++) outfile << corr1_orig[i]*LEN*DeltaT*DAcoupling*DAcoupling << endl;
    outfile.close();
    outfile.clear();
    */
    
    //[e'] Marcus and Marcus-Levich approximation
    double df= 1.0/LEN/DeltaT;
    double dE = df * 2 * pi;
    //outfile.open("Marcus-levich_EFGR_Jeff.dat");
    outfile.open((emptystr + "MarcusLevich_EFGR_Jeff_" + nameapp + ".dat").c_str());
    for (i=0; i<nn/2; i++) outfile << sqrt(pi/a_parameter_eff) * exp(-(dE*i*hbar-Er_eff)*(dE*i*hbar-Er_eff)/(4 * hbar*a_parameter_eff))*DAcoupling*DAcoupling << endl;
    outfile.close();
    outfile.clear();
    
    outfile.open((emptystr + "Marcus_EFGR_Jeff_" + nameapp + ".dat").c_str());
    for (i=0; i<nn/2; i++) outfile << sqrt(beta*pi/Er_eff) * exp(-beta*(dE*i*hbar-Er_eff)*(dE*i*hbar-Er_eff)/(4 * Er_eff))*DAcoupling*DAcoupling << endl;
    outfile.close();
    outfile.clear();
    
        
    case_count++;
    
    cout << "CASE # " << case_count <<  " done:" << endl;
    cout << "   beta = " << beta << endl;
    cout << "   eta = " << eta << endl;
    cout << "   Er_eff = " << Er_eff << endl;
    cout << "   a_parameter_eff = " << a_parameter_eff << endl;
    cout << "   df = " << df << endl;
    cout << "---------  ---------  ---------" << endl;
        
    }
    
    cout << "For all cases:" << endl;
    cout << "Omega =  " << Omega << endl;
    cout << "y0 = " << y_0 << endl;
    cout << "LEN = " << LEN << endl;
    cout << "DeltaT = " << DeltaT << endl;
    
    outfile2.close();
    outfile2.clear();
    
    /*
    // ********** BEGIN of Normal mode analysis ***********
    
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
    
    double omega_nm[n_omega]; //normal mode frequencies
    double req_nm[n_omega]; //req of normal modes (acceptor shift)
    double c_nm[n_omega];//coupling strength of normal modes
    double S_array[n_omega];//Huang-Rhys factor for normal modes
    double c_bath[n_omega];//secondary bath mode coupling coefficients
    
    double **D_matrix;// the Hessian matrix
    D_matrix = Create_matrix(n_omega, n_omega);
    double **TT_ns;
    TT_ns = Create_matrix(n_omega, n_omega);
    //transformation matrix: [normal mode]=[[TT_ns]]*[system-bath]
    //TT_ns * D_matrix * T_sn = diag, eigenvectors are row-vector of TT_ns
    double **Diag_matrix;
    Diag_matrix = Create_matrix(n_omega,n_omega);
    for (i=0; i < dim; i++) {
        for (j=0; j < dim; j++) {
            Diag_matrix[i][j] = 0;
        }
    }

    //calculating Er from discrete normal modes with ohmic bath modes. scanning eta
    //outfile1.open("Er_eta.dat");
    //for (eta = 1; eta <= 1 ; eta += 1)
    //{
    //eta=1.0;
    
    //cout << "c_bath[] = " << endl;
    //secondary bath mode min shift coefficients (for EXACT discrete normal mode analysis)
    Er_bath=0;
    for (w = 1; w < n_omega; w++) {
        //Ohmic SD:
        c_bath[w] = sqrt( 2.0 / pi * J_omega_ohmic(w*d_omega, eta) * d_omega * d_omega * w);
        //Gaussian SD:
        //c_bath[w] = sqrt( 2.0 / pi * S_omega_gaussian(w*d_omega, eta, sigma, omega_op) * d_omega * d_omega * w);
        Er_bath += 2.0 * c_bath[w] * c_bath[w] / (w*d_omega * w*d_omega);
        //cout << c_bath[w] << endl; //seems to be ok, proportional to sqrt(eta)
    }
    
    cout << "Er_bath = " << Er_bath << endl; //checked for eta linearality
    
    for (i=0; i< n_omega; i++) for (j=0; j<n_omega ;j++) D_matrix[i][j] = 0;
    D_matrix[0][0] = Omega*Omega;
    for (w =1 ; w < n_omega ; w++) {
        D_matrix[0][0] += pow(c_bath[w]/(w*d_omega) ,2);
        D_matrix[0][w] = D_matrix[w][0] = c_bath[w];
        D_matrix[w][w] = pow(w*d_omega ,2);
    }
    
    
    //cout << "Hession matrix D:" << endl;
    for (i=0; i < dim; i++) {
        for (j=0; j < dim; j++) {
            matrix[j][i] = D_matrix[i][j]; //switch i j to match with Fortran array memory index
            //cout << D_matrix[i][j] << " ";
        }
        //cout << endl;
    }
    
    
    //diagonalize matrix, the eigenvectors transpose is in result matrix => TT_ns.
    dsyev_('V', 'L', col, matrix[0], col, eig_val, work, lwork, info); //diagonalize matrix
    if (info != 0) cout << "Lapack failed. " << endl;
    
    for (i=0; i < dim; i++) omega_nm[i] = sqrt(eig_val[i]);
    
    //outfile.open("normal_mode_freq.dat");
    //for (i=0; i < dim; i++) outfile << omega_nm[i] << endl;
    //outfile.close();
    //outfile.clear();
    
    //cout << "eigen values = ";
    //for (i=0; i < dim; i++) cout << eig_val[i] <<"    ";
    //cout << endl;
    
    for (i=0; i < dim; i++)
        for (j=0; j < dim; j++) TT_ns[i][j] = matrix[i][j];
    
    // the coefficients of linear electronic coupling in normal modes (gamma[j]=TT_ns[j][0]*gamma_y), here gamma_y=1
    double gamma_nm[n_omega];
    for (i=0; i<n_omega; i++) gamma_nm[i] = TT_ns[i][0];
    
    //double shift_NE[n_omega]; //the s_j shifting for initial sampling
    //for (i=0; i<n_omega; i++) shift_NE[i] = s * gamma_nm[i];
    
    //req of normal modes (acceptor's potential energy min shift)
    for (i = 0; i < n_omega; i++) {
        req_nm[i] = 1 * TT_ns[i][0];
        for (a = 1; a < n_omega; a++) req_nm[i] -= TT_ns[i][a] * c_bath[a] / (a*d_omega * a*d_omega);
    }
    
    //cout << "normal mode req_nm:" << endl;
    //outfile.open("Huang-Rhys_nm.dat");
    for (i = 0; i < n_omega; i++) {
        //tilde c_j coupling strength normal mode
        c_nm[i] = req_nm[i] * omega_nm[i] * omega_nm[i];
        //cout << c_nm[i] << endl;
        
        req_nm[i] *= 2.0 * y_0;
        //cout << req_nm[i] << endl;
        //discrete Huang-Rhys factor as follows
        S_array[i] = omega_nm[i] * req_nm[i] * req_nm[i] * 0.5;
        //outfile << S_array[i] << endl;
    }
    //outfile.close();
    //outfile.clear();
    
    // ******** END of Normal mode analysis **************
    
    //the exact reorganization energy Er for Marcus theory from normal modes
    Er = 0;
    a_parameter = 0;
    
    //for (i = 0; i < n_omega; i++) Er += 2.0 * c_nm[i] * c_nm[i] / (omega_nm[i] * omega_nm[i]);
    for (i = 0; i < n_omega; i++) Er += 0.5 * omega_nm[i] * omega_nm[i] * req_nm[i] * req_nm[i]; //S_array[i] * omega_nm[i];
    
    for (i = 0; i < n_omega; i++) a_parameter += 0.5 * S_array[i] * omega_nm[i] * omega_nm[i] /tanh(beta*hbar* omega_nm[i] *0.5);
    
    outfile1 << Er << endl;
    cout << "Er (discrete) = " << Er << endl;
    cout << "a_parameter (discrete) = "<< a_parameter << endl;
    //}
    outfile1.close();
    outfile1.clear();
    
    
    
    //===========case: [2] Eq FGR using discreitzed SD J_o(\omega) exact ===========
    outfile1.open("Integral_nm_exact.dat");
    for (i = 0; i < nn; i++) corr1[i] = corr2[i] = 0; //zero padding
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
        }
        integral_re = Sum(integ_re, n_omega);
        integral_im = Sum(integ_im, n_omega);
        
        outfile1 << integral_re << "\t" << integral_im << endl;
        
        corr1[i] = exp(-1 * integral_re) * cos(integral_im);
        corr2[i] = -1 * exp(-1 * integral_re) * sin(integral_im);
    }
    
    FFT(-1, mm, corr1, corr2);//notice its inverse FT
    
    for(i=0; i<nn; i++) { //shift time origin
        corr1_orig[i] = corr1[i] * cos(2*pi*i*shift/N) - corr2[i] * sin(-2*pi*i*shift/N);
        corr2_orig[i] = corr2[i] * cos(2*pi*i*shift/N) + corr1[i] * sin(-2*pi*i*shift/N);
    }
    
    outfile.open("Exact_EFGR_nm.dat");
    for (i=0; i<nn/2; i++) outfile << corr1_orig[i]*LEN*DeltaT*DAcoupling*DAcoupling << endl;
    outfile.close();
    outfile.clear();
    outfile1.close();
    outfile1.clear();
     */

     
    //-------------- Summary ----------------
    
    cout << "-----THERMAL CONDITION------- " << endl;
    cout << "Omega (primary) = " << Omega << endl;
    cout << "normal modes n_omega = " << n_omega << endl;
    cout << "omega_max = " << omega_max << endl;
    cout << "d_omega_eff = " << d_omega_eff << endl;
    cout << "beta = " << beta << endl;
    cout << "eta  = " << eta << endl;
    //cout << "initial shift s = " << s << endl;
    cout << "--------- END of EFGR in Condon case --------" << endl;
    
    
    
    return 0;
}



/********* SUBROUTINE *************/


//spectral densities

double S_omega_ohmic(double omega, double etaa) {
    return etaa * omega * exp(-1 * omega / omega_c);
}

double S_omega_drude(double omega, double etaa) {
    return etaa * omega /(1 + omega*omega);
}

double S_omega_gaussian(double omega, double etaa, double sigma, double omega_op) {
    return   0.5 / hbar * etaa * omega * exp(-(omega - omega_op)*(omega - omega_op)/(2*sigma*sigma))/RT_2PI/sigma;
}

double J_omega_ohmic(double omega, double etaa) {
    //notice definition J(omega) is different from S(omega)
    //J_omega = pi/2 * sum_a c_a^2 / omega_a delta(omega - omega_a)
    return etaa * omega * exp(-1 * omega / omega_c);
}

double J_omega_ohmic_eff(double omega, double etaa) {
    //(normal mode) effective SD for Ohmic bath DOF
    //J_omega = pi/2 * sum_a c_a^2 / omega_a delta(omega - omega_a)
    return etaa * omega * pow(Omega,4) *y_0*y_0 / ( pow(Omega*Omega - omega*omega, 2) + etaa*etaa*omega*omega);
}

//min-to-min energy as Fourier transform frequency
void Integrand_LSC(double omega, double t, double &re, double &im) {
    re = (1-cos(omega*t))/tanh(beta*hbar*omega/2);
    im = sin(omega*t);
    return;
}

void Integrand_LSC_inh(double omega, double t, double &re, double &im) {
    re = omega*omega*t*t/2/tanh(beta*hbar*omega/2);
    im = omega*t;
    return;
}

void Integrand_CL_avg(double omega, double t, double &re, double &im) {
    re = (1-cos(omega*t))*2/(beta*hbar*omega);
    im = sin(omega*t);
    return;
}

void Integrand_CL_donor(double omega, double t, double &re, double &im) {
    re = (1-cos(omega*t))*2/(beta*hbar*omega);
    im = omega*t;
    return;
}

void Integrand_2cumu(double omega, double t, double &re, double &im) {
    re = (1-cos(omega*t))*2/(beta*hbar*omega);
    im = omega*t;
    return;
}

void Integrand_2cumu_inh(double omega, double t, double &re, double &im) {
    re = omega*t*t/(beta*hbar);
    im = omega*t;
    return;
}

//noneq FGR

void Integrand_NE_exact(double omega, double tp, double tau, double shift, double req, double &re, double &im) {//including Huang-Rhys factor S_j
    re = omega*req*req*0.5*(1-cos(omega*tau))/tanh(beta*hbar*omega/2);
    im = omega*req*req*0.5*sin(omega*tau) + omega*req*shift* (sin(omega*tp) - sin(omega*tp-omega*tau));
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

double Integrate_from(double *data, int sp, int n, double dx){
    double I(0);
    I += (data[sp]+data[n-1])/2;
    for (int i=sp+1; i< n-1; i++) {
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


void DFT(int dir, int m, double *x, double *y) {
    /*
     This code computes an in-place complex-to-complex DFT by direct approach
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
    int n,i,j,k,N;
    
    // Calculate the number of points
    N = 1;
    for (i=0;i<m;i++)
        N *= 2;
    
    double *re = new double [N];
    double *im = new double [N];
    for (n = 0; n < N; n++) re[n] = im[n] = 0;
    double w = 2 * pi / N;
    
    
    if (abs(dir) != 1 ) cout << "error from DFT subroutine: dir ill defined"<< endl;
    
    for (n=0; n<N; n++) {
        for (k=0; k<N; k++) {
            re[n] += x[k] * cos(w*n*k) + dir * y[k] * sin(w*n*k);
            im[n] += y[k] * cos(w*n*k) - dir * x[k] * sin(w*n*k);
        }
    }
    
    for (n=0; n<N; n++) {
        x[n] = re[n];
        y[n] = im[n];
    }
    
    if (dir == -1)
        for (n=0; n<N;n++) {
            x[n] /= N;
            y[n] /= N;
        }
    
    delete [] re;
    delete [] im;
    return;
}

double** Create_matrix(int row, int col) {
    double **matrix = new double* [col];
    matrix[0] = new double [col*row];
    for (int i = 1; i < col; ++i)
        matrix[i] = matrix[i-1] + row;
    return matrix;
}



