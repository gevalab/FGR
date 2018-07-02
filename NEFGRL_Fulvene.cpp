/* This code calculates Non-equilibrium Fermi Golden Rule rate 
   in non-Condon case using Brownian oscillator model
   compare with linearized semiclassical methods -- using LVC model
   To compile: g++ -o NEFGRL NEFGRL.cpp
   (c) Xiang Sun 2015
 */

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
using namespace std;

double temperature = 300; //K
const double tp_max_fs = 50; //fs
const double Deltatp_fs = 0.25; //0.25 time step of tp (fs)
const int DTauPerDtp = 5;//100; //# of DeltaTau per Deltatp
const double s = -1; //Noneq. initial shift

//Fulvene molecule LVC Hamiltonian (a.u.)
const int n_omega = 30; //number of normal modes
double Delta_11 = 0; //donor
double Delta_22 = 1.8764e-1; //acceptor
double omega_array[n_omega] = {
    0.00098662,
    0.0017237,
    0.002419,
    0.0028634,
    0.0033546,
    0.0034833,
    0.0036851,
    0.0038835,
    0.0043459,
    0.0041842,
    0.0043828,
    0.0044617,
    0.0046002,
    0.0051482,
    0.005131,
    0.0057261,
    0.0057567,
    0.0067668,
    0.007755,
    0.0072129,
    0.0076025,
    0.007927,
    0.0082881,
    0.0085797,
    0.016669,
    0.017018,
    0.017046,
    0.017134,
    0.017176,
    0.017397
};
double d2_array[n_omega] = {
    0,
    -4.017e-10,
    0,
    0,
    0.000376,
    0,
    0,
    0,
    1.9402e-08,
    0,
    0,
    0,
    0.00048475,
    2.8393e-09,
    0.00046551,
    1.4564e-07,
    -0.00056739,
    -5.3006e-08,
    9.9417e-09,
    0.00061246,
    0.0004844,
    -0.0016541,
    6.3709e-08,
    0.0024292,
    -0.00047442,
    -6.1685e-08,
    9.7464e-05,
    -9.6004e-08,
    -8.4104e-05,
    1.5356e-08
};

double c_array[n_omega] = {
    0,
    -9.2519e-06,
    0,
    0,
    -2.0893e-10,
    0,
    0,
    0,
    0.00043069,
    0,
    0,
    0,
    -1.0211e-08,
    0.00037411,
    -2.1549e-09,
    4.573e-05,
    9.7768e-09,
    0.00051583,
    0.0009486,
    3.1798e-08,
    1.2354e-08,
    -2.0903e-09,
    -0.00035004,
    -3.9514e-09,
    2.0579e-10,
    7.0815e-06,
    -3.3803e-09,
    -5.233e-05,
    1.0461e-08,
    3.0324e-06
};

/*
double d2_mctdh[n_omega] = {
    0,
    -9.67550e-09,
    0,
    0,
    0.006491800,
    0,
    0,
    0,
    2.94310e-07,
    0,
    0,
    0,
    0.0071471,
    3.957200e-08,
    0.00649880,
    1.92460e-06,
    -0.00747810,
    -6.443700e-07,
    1.128900e-07,
    0.0072115,
    0.0055555,
    -0.018578,
    6.997900e-07,
    0.0262260,
    -0.0036746,
    -4.72850e-07,
    0.000746500,
    -7.334300e-07,
    -0.00064173,
    1.16430e-07
};
double c_mctdh[n_omega] = {
    0,
    -0.00022284,
    0,
    0,
    -3.60720e-09,
    0,
    0,
    0,
    0.0065332,
    0,
    0,
    0,
    -1.50550e-07,
    0.0052140,
    -3.008300e-08,
    0.000604330,
    1.288600e-07,
    0.0062708,
    0.0107720,
    3.744100e-07,
    1.416900e-07,
    -2.34770e-08,
    -0.0038450,
    -4.26600e-08,
    1.593900e-09,
    5.428500e-05,
    -2.58900e-08,
    -0.00039978,
    7.981800e-08,
    2.299100e-05
};
 */

const double fs2au = 41.341105; //1 fs = 41.34 a.u.
const double kT2au = 3.168429139e-6;// kT(1 Kelvin) = 3.2e-6 a.u.

double beta = 1.0 / (temperature * kT2au); //in a.u.
double tp_max = tp_max_fs * fs2au; //a.u.
double Deltatp = Deltatp_fs * fs2au; //a.u. time step of tp
double DeltaTau = Deltatp / DTauPerDtp;//50; //a.u. time step of tau (inner loop)


const double DAcoupling = 1;
const double pi=3.14159265358979324;
const double RT_2PI= sqrt(2*pi);
const double hbar = 1;


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



int main (int argc, char *argv[]) {
    
    stringstream ss;
    string emptystr("");
    string filename;
    string idstr("");
    string nameapp;

    int i, j, a, b;
    double omega;
    int w; //count of omega
    double integ_re[n_omega];
    double integ_im[n_omega];
    
    ofstream outfile;
    ofstream outfile1;
    
    double integral_re, integral_im;
    integral_re = integral_im = 0;

    
    int M; //time slice for tau = 0 --> tp
    int m; //index of tau
    
    double linear_accum_re;
    double linear_accum_im;
    double linear_re;
    double linear_im;
    double temp_re;
    double temp_im;
    double C_re;
    double C_im;
    
    double omega_nm[n_omega]; //normal mode frequencies
    double req_nm[n_omega] = {
        -0,
        0.0001352,
        -0,
        -0,
        -33.413,
        -0,
        -0,
        -0,
        -0.0010273,
        -0,
        -0,
        -0,
        -22.907,
        -0.00010713,
        -17.682,
        -0.0044417,
        17.121,
        0.0011576,
        -0.00016531,
        -11.772,
        -8.3809,
        26.323,
        -0.00092745,
        -33,
        1.7074,
        0.000213,
        -0.33541,
        0.00032702,
        0.28509,
        -5.0739e-05
    }; //req of normal modes (acceptor shift)
    double gamma_nm[n_omega]; //linear coupling coefficient
    double shift_NE[n_omega]; //the s_j shifting for initial sampling
    
    double tp; //t' for noneq preparation
    double tau;
    
    double kre, kim;
    double sum(0);
    double kneq(0);
    
    cout << "---- BEGIN of NEFGRL in non-Condon case for Fulvene ----" << endl;
    
    double dE12 = -0.98911; //Delta22 - 0.5* sum d^2/w^2
    double omega_DA = -dE12; //fixed omega_DA, with scan tp
    
    omega_DA = Delta_22;
    for (i=0; i<n_omega ;i++) {
        omega_DA -= d2_array[i] * d2_array[i] / (2 * omega_array[i] * omega_array[i]);
    }
    cout << "omega_DA (a.u.) = " << omega_DA << endl;
    
    
    ss.str("");
    nameapp = "";
    ss << "Fulvene_"<< temperature << "K";
    nameapp = ss.str();

    for (i=0; i<n_omega; i++) omega_nm[i] = omega_array[i];
    //prepare modes: for noneq initial shift and coupling
    //req of normal modes (acceptor's potential energy min shift)
    
    for (i=0; i<n_omega; i++) {
        req_nm[i] = - d2_array[i] / (omega_array[i]*omega_array[i]);
    }
    
    for (i=0; i<n_omega; i++) gamma_nm[i] = c_array[i];
    
    for (i=0; i<n_omega; i++) shift_NE[i] = s * req_nm[i]; //initial shift (from bottom of A)

    double Er=0;
    for (i = 0; i < n_omega; i++) Er += 0.5 * omega_nm[i] * omega_nm[i] * req_nm[i] * req_nm[i];
    cout << "Er = " << Er << endl;
    /*
    ss.str("");
    idstr = "";
    ss << "s" << s;
    ss << "w" << omega_DA ;
    idstr += ss.str();
    */
    
    
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
   
    


    //-------------- Summary ----------------

    cout << "---- SUMMARY ---- " << endl;
    cout << "   Er = " << Er << endl;
    cout << "   Temperature(K) = " << temperature << endl;
    cout << "   beta (a.u.) = " << beta << endl;
    cout << "   fix omega_DA (a.u.) = " << omega_DA << endl;
    cout << "   Delta tp (fs) = " << Deltatp_fs << endl;
    cout << "   # of DeltaTau per Deltatp = " << DTauPerDtp << endl;
    cout << "   Total number of tp = " << tp_max_fs/Deltatp_fs <<  endl;
    cout << "   normal modes n_omega = " << n_omega << endl;
    cout << "   initial shift s = " << s << endl;
    cout << "---- END of NEFGRL in non-Condon case ----" << endl;
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

/*
void Integrand_NE_exact(double omega, double tp, double tau, double shift, double req, double &re, double &im) {//including Huang-Rhys factor S_j
    re = omega*req*req*0.5*(-1-cos(omega*tau))/tanh(beta*hbar*omega/2); //WRONG one: "-1"
    im = omega*req*req*0.5*sin(omega*tau) + omega*req*shift* (sin(omega*tp) + sin(omega*tau - omega*tp));
    return;
}
 */


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







