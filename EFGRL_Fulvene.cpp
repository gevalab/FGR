/* This code calculates equilibrium Fermi Golden Rule rate
   in non-Condon case using LVC model
   compare with linearized semiclassical methods 
   To compile: g++ -o EFGRL EFGRL.cpp
   (c) Xiang Sun 2016
 */

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
using namespace std;

//molecule LVC Hamiltonian (a.u.)
string molecule = "Fulvene"; //Fulvene, BMA or MIA
const int n_omega = 30;//78; //number of normal modes, Fulvene=30; BMA=78; MIA=96
double Delta_11 = 0; //donor
double Delta_22 = -0.98911;//acceptor. Fulvene=-0.98911; BMA=2.9255e-02; MIA=2.4615e-03(reversed)
const double fs2au = 41.341105; //1 fs = 41.34 a.u.
const double kT2au = 3.168429139e-6;// kT(1 Kelvin) = 3.2e-6 a.u.

double temperature = 4000; //K
const int LEN = 1024; //FFT time steps
double DeltaT_fs = 0.04;
double DeltaT = DeltaT_fs * fs2au;
double T0 = -DeltaT*LEN/2;

double beta = 1.0 / (temperature * kT2au); //in a.u.
const double DAcoupling = 1;
const double pi=3.14159265358979324;
const double RT_2PI= sqrt(2*pi);
const double hbar = 1;
double tdamp = 50 * fs2au; //damping exponential
double gdamp = 6400 * fs2au * fs2au; //gaussian damping


void FFT(int dir, int m, double *x, double *y); //Fast Fourier Transform, 2^m data
double** Create_matrix(int row, int col);//new continuous 2D array in heap
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



int main (int argc, char *argv[]) {

    stringstream ss;
    string emptystr("");
    string filename("");
    string idstr("");
    string nameapp("");
    
    int i, j, a, b;
    double d2_mctdh[n_omega];//original MCTDH d2
    double c_mctdh[n_omega]; //original MCTDH c
    double d2_array[n_omega];//LVC d2
    double c_array[n_omega]; //LVC c
    double omega_nm[n_omega]; //normal mode frequencies (a.u.)
    double req_nm[n_omega]; //req of normal modes (acceptor shift)
    double gamma_array[n_omega]; //linear coupling coefficient
    double S_array[n_omega];//Huang-Rhys factor
    //double shift_NE[n_omega]; //the s_j shifting for initial sampling
    
    //input of parameters
    ifstream infile;
    infile.open((emptystr+ "omega_"+ molecule+".txt").c_str());
    if (!infile.is_open()) {
        cout << "error: input file omega cannot open"<< endl;
        return -1;
    }
    for (i=0; i< n_omega; i++) infile >> omega_nm[i];
    infile.close();
    infile.clear();
    
    infile.open((emptystr+ "d2_mctdh_"+ molecule+".txt").c_str());
    if (!infile.is_open()) {
        cout << "error: input file d2 cannot open"<< endl;
        return -1;
    }
    for (i=0; i< n_omega; i++) infile >> d2_mctdh[i];
    infile.close();
    infile.clear();
    
    infile.open((emptystr+ "c_mctdh_"+ molecule+".txt").c_str());
    if (!infile.is_open()) {
        cout << "error: input file c cannot open"<< endl;
        return -1;
    }
    for (i=0; i< n_omega; i++) infile >> c_mctdh[i];
    infile.close();
    infile.clear();
    
    cout << "----> BEGIN of EFGRL in non-Condon case of " << molecule << endl;
    
    //scale parameters to LVC model
    for (i=0; i< n_omega; i++) d2_array[i] = d2_mctdh[i] * sqrt(omega_nm[i]);
    for (i=0; i< n_omega; i++) c_array[i] = c_mctdh[i] * sqrt(omega_nm[i]);
    
    //then convert LVC to our model
    //prepare modes: for noneq initial shift and coupling
    //req of normal modes (acceptor's potential energy min shift)
    for (i=0; i< n_omega; i++) req_nm[i] = -1 * d2_array[i]/(omega_nm[i]*omega_nm[i]);
    for (i=0; i< n_omega; i++) gamma_array[i] = c_array[i];
    //for (i=0; i< n_omega; i++) shift_NE[i] = shift * req_nm[i]; //shift = -1, from min of A

    double dE = 0; // = - (Delta22 - 0.5* sum d^2/w^2) = - hbar * omega_DA
    double omega_DA = 0; // -dE = hbar * omega_DA
    
    omega_DA = Delta_22 - Delta_11;
    for (i=0; i< n_omega ;i++) {
        omega_DA -= d2_array[i] * d2_array[i] / (2 * omega_nm[i] * omega_nm[i]);
    }
    //MARK: for MIA only: omega_{DA = - omega_DA;
    cout << "omega_DA (a.u.) = " << omega_DA << endl;
    
    //reorganization energy
    double Er=0;
    for (i=0; i< n_omega; i++) Er += 0.5 * omega_nm[i] * omega_nm[i] * req_nm[i] * req_nm[i];
    cout << "Er = " << Er << endl;
    
    
    int mm(0), nn(1); // nn = 2^mm is number of (complex) data to FFT
    
    while (nn < LEN ) {
        mm++;
        nn *= 2;
    } //nn is the first 2^m that larger LEN
    
    double *corr1 = new double [nn];
    double *corr2 = new double [nn];
    double *corr1_orig = new double [nn]; //shifted origin to T0
    double *corr2_orig = new double [nn];
    double *integrand = new double [n_omega];
    double t;
    double integral_re, integral_im;
    double linear_accum_re;
    double linear_accum_im;
    double linear_re;
    double linear_im;
    double temp_re;
    double temp_im;
    double C_re;
    double C_im;
    double kre, kim;
    double omega;
    int w; //count of omega
    double integ_re[n_omega];
    double integ_im[n_omega];
    double shift = T0 / DeltaT;
    double N = nn;
    ofstream outfile;
    ofstream outfile1;
    
    
    for (i=0; i <n_omega; i++) {
        //discrete Huang-Rhys factor
        S_array[i] = omega_nm[i] * req_nm[i] * req_nm[i] * 0.5;
    }
    

    ss.str("");
    nameapp = "";
    ss << molecule << "_"<< temperature << "K";
    nameapp = ss.str();

    
    // Calculate spectral density by histogram
    cout << "---- Histogram Spectral Density ---- " << endl;
    const int MAXBIN = n_omega; //for histogramming SD
    double influence[MAXBIN]; //discrete SD histogram (influence density of states)=J(omega)
    int bin;
    int extreme_count(0);
    double min_val = 0;
    double omega_max_hist = omega_nm[n_omega-1]*1.01;//the largest freq
    double dr = (omega_max_hist - min_val)/MAXBIN; //bin size
    double J_array[n_omega];//J_j = pi/2 * c_j^2 / omega_j = pi/8 * omega^3 * req^2
    
    //discrete J spectral density
    for (i = 0; i < n_omega; i++) {
        J_array[i] = pi /8.0 * pow(omega_nm[i],3) * req_nm[i]* req_nm[i];
        // J_array[i] = c_nm[i] * c_nm[i] / omega_nm[i] * pi * 0.5;
    }
    
    for (bin = 0; bin < MAXBIN; bin++) influence[bin] = 0.0;
    
    for (i = 0; i < n_omega; i++) {
        bin = static_cast<int> ((omega_nm[i] - min_val)/dr);//((omega_nm[i]+dr*0.5-min_val)/dr);
        if (bin >=0 && bin < MAXBIN) influence[bin] += J_array[i]; //histogramming
        else extreme_count++;
    }
    if (extreme_count != 0)
        cout << "Total extreme histogram points: " << extreme_count << endl;
    
    for (bin = 0; bin < MAXBIN; bin++) influence[bin] /= (n_omega-extreme_count);
    // for multiple configurations need to /hist_count
    
    //normalization
    double sum_J(0);
    double sum_influence(0);
    for (bin = 0; bin < MAXBIN; bin++) sum_influence += influence[bin];
    
    for (i = 0; i < n_omega; i++) sum_J += J_array[i];
    
    for (bin = 0; bin < MAXBIN; bin++) influence[bin] *= sum_J / (dr * sum_influence);
    
    outfile.open((emptystr + "J_nm_hist_" + nameapp + ".dat").c_str());
    for (bin = 0; bin < MAXBIN; bin++) outfile << influence[bin] << endl;
    outfile.close();
    outfile.clear();
    cout << "omega_max_hist = " << omega_max_hist << endl;
    cout << "MAXBIN = " << MAXBIN << endl;
    
    
    /*
    ss.str("");
    idstr = "";
    ss << "s" << s;
    ss << "w" << omega_DA ;
    idstr += ss.str();
    */
    
    
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
    /*
    outfile.open((emptystr + "Exact_EFGRL_tre_" + nameapp + ".dat").c_str());
    for(i=0; i<nn; i++) outfile << corr1[i] << endl;
    outfile.close();
    outfile.clear();
    
    outfile.open((emptystr + "Exact_EFGRL_tim_" + nameapp + ".dat").c_str());
    for(i=0; i<nn; i++) outfile << corr2[i] << endl;
    outfile.close();
    outfile.clear();
     
    
    // add damping factor for filtering tail out
    for (i = 0; i < LEN; i++) {
        t = T0 + DeltaT * i;
        corr1[i] *= exp(- abs(t) / tdamp);
        corr2[i] *= exp(- abs(t) / tdamp);
    }
    */
    
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
    
    /*
    outfile.open((emptystr + "LSC_EFGRL_tre_" + nameapp + ".dat").c_str());
    for(i=0; i<nn; i++) outfile << corr1[i] << endl;
    outfile.close();
    outfile.clear();
    
    outfile.open((emptystr + "LSC_EFGRL_tim_" + nameapp + ".dat").c_str());
    for(i=0; i<nn; i++) outfile << corr2[i] << endl;
    outfile.close();
    outfile.clear();
    
    
    // add damping factor for filtering tail out
    for (i = 0; i < LEN; i++) {
        t = T0 + DeltaT * i;
        corr1[i] *= exp(- abs(t) / tdamp);
        corr2[i] *= exp(- abs(t) / tdamp);
    }
     */
    
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
    
    /*
    // add damping factor for filtering tail out
    for (i = 0; i < LEN; i++) {
        t = T0 + DeltaT * i;
        corr1[i] *= exp(- abs(t) / tdamp);
        corr2[i] *= exp(- abs(t) / tdamp);
    }
    */
    
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
    
    
    // add damping factor for filtering tail out
    for (i = 0; i < LEN; i++) {
        t = T0 + DeltaT * i;
        corr1[i] *= exp(-t*t / gdamp); //exp(- abs(t) / tdamp* 1.7);
        corr2[i] *= exp(-t*t / gdamp); //exp(- abs(t) / tdamp* 1.7);
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
    
    
    // add damping factor for filtering tail out
    for (i = 0; i < LEN; i++) {
        t = T0 + DeltaT * i;
        corr1[i] *= exp(-t*t / gdamp); //exp(- abs(t) / tdamp * 2.4);
        corr2[i] *= exp(-t*t / gdamp); //exp(- abs(t) / tdamp * 2.4);
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
   
    


    //-------------- Summary ----------------

    cout << "---- SUMMARY ---- " << endl;
    cout << "   Er = " << Er << endl;
    cout << "   Temperature(K) = " << temperature << endl;
    cout << "   beta (a.u.) = " << beta << endl;
    cout << "   omega_DA (a.u.) = " << omega_DA << endl;
    cout << "   # of normal modes, n_omega = " << n_omega << endl;
    cout << "   LEN  = " << LEN << endl;
    cout << "   DeltaT (fs)  = " << DeltaT_fs << endl;
    cout << "---- END of EFGRL in non-Condon case ----" << endl;
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







