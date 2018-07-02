/* This code calculates Non-equilibrium Fermi Golden Rule rate 
   in Condon case using Brownian oscillator model 
   compare with linearized semiclassical methods  
   To compile: g++ -o NEFGR NEFGR.cpp -llapack -lrefblas -lgfortran
   (c) Xiang Sun 2015
 */

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
using namespace std;

// *********** PARAMETERS *************
double omega_DA_fix = 0; //fixed omega_DA, with scan tp
double s = -3; //Noneq. initial shift of parimary mode
double Omega = 0.5; //primary mode freq
double y_0 = 1; //shift of primary mode
const int n_omega = 200; // total num of modes
const double omega_max = 15;//20;//2.5 for gaussian// 20 for ohmic
const double DeltaTau =0.002; //time slice for t' griding
const double tp_max = 20; //scanning tp option, DeltaTau as step
const double Deltatp = 0.2;
const int bldim = 3;
const int eldim = 3;
double beta_list[bldim] = {1, 2, 5};//{1};//{1, 2, 5};//{0.2, 1.0, 5.0};
double eta_list[eldim] = {0.5, 1, 5};//{1};//{0.5, 1, 5};//{0.2, 1.0, 5.0};
// ************************************

double beta = 0.2;//0.2;//1;//5;
double eta  = 0.5; //0.2;//1;//5;
const double DAcoupling = 0.1;
double tp_fix = 5; //fixed t' for noneq FGR rate k(t',omega_DA) with scan omega_DA
const double d_omega = omega_max / n_omega;//0.1;//0.002;for gaussian//0.1; for ohmic
const double d_omega_eff = omega_max / n_omega; //for effective SD sampling rate

const int LEN = 1024;//512;//1024;
const double DeltaT=0.1;//0.2; //FFT time sampling interval
const double T0= -DeltaT*(LEN*0.5);//-DeltaT*LEN/2+DeltaT/2;

const double pi=3.14159265358979324;
const double RT_2PI= sqrt(2*pi);
const double hbar = 1;
//for gaussian spectral density
const double sigma = 0.1;
const double omega_op = 1.0;

void FFT(int dir, int m, double *x, double *y); //Fast Fourier Transform, 2^m data
void DFT(int dir, int m, double *x, double *y); //Discrete Fourier Transform
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
double Integrate(double *data, int n, double dx);
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
    
    ofstream outfile;// k(t')
    ofstream outfile1;//P(t) the noneq probability of being on the donor state
    ofstream outfile2; //keq at omega_DA_fix
    
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
    double *tau_array = new double [M];
    double *C_re = new double [M];
    double *C_im = new double [M];
    double d_omega_DA = 2 * pi / LEN / DeltaT; //omega_DA griding size
    double omega_DA;
    double kre, kim;
    double sum(0);
    double kneq(0);
    
    int beta_index(0);
    int eta_index(0);
                  
    cout << "-------------- NEFGR in Condon case --------------" << endl;
    
    ss.str("");
    idstr = "";
    ss << "w" << omega_DA_fix ;
    idstr += ss.str();
    //keq of QMLSC in Condon case
    outfile2.open((emptystr + "k_QMLSC_EFGR_" + nameapp + idstr + ".dat").c_str());
    
    
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

    
    //setting up spectral density
    for (w = 0; w < n_omega; w++) J_eff[w] = J_omega_ohmic_eff(w*d_omega_eff, eta);
    for (w = 0; w < n_omega; w++) SD[w] = S_omega_ohmic(w*d_omega, eta); //Ohmic spectral density
    //for (w = 0; w < n_omega; w++) SD[w] = S_omega_drude(w*d_omega, eta); //Drude spectral density
    //for (w = 0; w < n_omega; w++) SD[w] = S_omega_gaussian(w*d_omega, eta, sigma, omega_op);
    
    //outfile.open("S(omega).dat");
    //for (i=0; i< n_omega; i++) outfile << SD[i] << endl;
    //outfile.close();
    //outfile.clear();
    /*
    outfile.open("J_eff(omega).dat");
    for (i=0; i< n_omega; i++) outfile << J_eff[i] << endl;
    outfile.close();
    outfile.clear();
    */
    
    
    /*
    double integrand[n_omega];
    for (w = 0; w < n_omega; w++) integrand[w] = SD[w] * w *d_omega;
    integrand[0]=0;
    Er = Integrate(integrand, n_omega, d_omega);
    
    for (w = 1; w < n_omega; w++) integrand[w] = SD[w] * w*d_omega * w*d_omega /tanh(beta*hbar* w * d_omega*0.5);
    integrand[0]=0;
    a_parameter = 0.5 * Integrate(integrand, n_omega, d_omega);
    //cout << "Er = " << Er << endl;
    //cout << "a_parameter = " << a_parameter << endl;
    */
  
    for (w = 1; w < n_omega; w++) req_eff[w] = sqrt(8 * hbar * J_eff[w] / (pi * w * d_omega_eff*w * d_omega_eff*w));//eq min for each eff normal mode
    
    double c_bath[n_omega]; //secondary bath mode min shift coefficients
    for (w = 1; w < n_omega; w++) {
        c_bath[w] = sqrt( 2 / pi * J_omega_ohmic(w*d_omega, eta) * d_omega * d_omega * w);
    }
    
    //********** BEGIN of Normal mode analysis ***********
    
    //construct Hessian matrix
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
            matrix[j][i] = D_matrix[i][j];
            //NOTE: switch i j to match with Fortran array memory index
            //cout << D_matrix[i][j] << " ";
        }
        //cout << endl;
    }
    
    //diagonalize matrix, the eigenvectors transpose is in result matrix => TT_ns.
    dsyev_('V', 'L', col, matrix[0], col, eig_val, work, lwork, info); //diagonalize matrix
    if (info != 0) cout << "Lapack failed. " << endl;
    
    for (i=0; i < dim; i++) omega_nm[i] = sqrt(eig_val[i]);
    
    /*
    outfile.open("normal_mode_freq.dat");
    for (i=0; i < dim; i++) outfile << omega_nm[i] << endl;
    outfile.close();
    outfile.clear();
     */
    
    //cout << "eigen values = ";
    //for (i=0; i < dim; i++) cout << eig_val[i] <<"    ";
    //cout << endl;
    
    for (i=0; i < dim; i++)
        for (j=0; j < dim; j++) TT_ns[i][j] = matrix[i][j];
    
    /*
     cout << "diagonalized Hessian matrix: " << endl;
     for (i=0; i < dim; i++) {
     for (j=0; j < dim; j++) {
     for (a=0; a < dim; a++)
     for (b=0; b < dim; b++) Diag_matrix[i][j] += TT_ns[i][a]*D_matrix[a][b]*TT_ns[j][b]; //TT_ns * D * T_sn
     cout << Diag_matrix[i][j] << "    " ;
     }
     cout << endl;
     }
     
     cout << endl;
     cout << "transformation matrix TT_ns (TT_ns * D * T_sn = diag, eigenvectors are row-vector of TT_ns): " << endl;
     for (i=0; i < dim; i++) {
     for (j=0; j < dim; j++) cout << TT_ns[i][j] << "    " ;
     cout << endl;
     }
     */
    
    // the coefficients of linear electronic coupling in normal modes (gamma[j]=TT_ns[j][0]*gamma_y), here gamma_y=1
    
    //Noneq initial shift of each mode
    for (i=0; i<n_omega; i++) gamma_nm[i] = TT_ns[i][0];
    for (i=0; i<n_omega; i++) shift_NE[i] = s * gamma_nm[i];
    
    
    //req of normal modes (acceptor's potential energy min shift)
    for (i=0; i<n_omega; i++) {
        req_nm[i] = 1 * TT_ns[i][0];
        for (a=1; a < n_omega; a++) req_nm[i] -= TT_ns[i][a] * c_bath[a] / (a*d_omega * a*d_omega);
    }
    
    //outfile.open("Huang-Rhys.dat");
    for (i=0; i<n_omega; i++) {
        //tilde c_j coupling strength normal mode
        c_nm[i] = req_nm[i] * omega_nm[i] * omega_nm[i];
        req_nm[i] *= 2 * y_0;
        //discrete Huang-Rhys factor
        S_array[i] = omega_nm[i] * req_nm[i] * req_nm[i] /2;
        //outfile << S_array[i] << endl;
    }
    outfile.close();
    outfile.clear();
    
    //******** END of Normal mode analysis **************
    
    //Case [1]: Equilibrium exact QM / LSC in Condon case using discreitzed J(\omega)
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
            corr1[i] = exp(-1 * integral_re) * cos(integral_im);
            corr2[i] = -1 * exp(-1 * integral_re) * sin(integral_im);
    }
    
    FFT(-1, mm, corr1, corr2);//notice its inverse FT
     
    for(i=0; i<nn; i++) { //shift time origin
         corr1_orig[i] = corr1[i] * cos(2*pi*i*shift/N) - corr2[i] * sin(-2*pi*i*shift/N);
         corr2_orig[i] = corr2[i] * cos(2*pi*i*shift/N) + corr1[i] * sin(-2*pi*i*shift/N);
    }
     
    outfile.open((emptystr + "QMLSC_EFGR_" + nameapp + ".dat").c_str());
    for (i=0; i<nn/2; i++) outfile << corr1_orig[i]*LEN*DeltaT*DAcoupling*DAcoupling << endl;
    outfile.close();
    outfile.clear();
    //output keq at omega_DA_fix
    i = static_cast<int> (omega_DA_fix/d_omega_DA+0.5);
    outfile2 << corr1_orig[i]*LEN*DeltaT*DAcoupling*DAcoupling << endl;


    //Case [2]: Noneq exact QM / LSC in Condon case using discreitzed J(\omega)

    //option [A]: fix tp, scan omega_DA
    /*
    for (m=0; m < M; m++) {//tau index
        tau = m * DeltaTau;
        integ_re[0] = 0;
        integ_im[0] = 0;
        for (w = 0; w < n_omega; w++) {
            Integrand_NE_exact(omega_nm[w], tp, tau, shift_NE[w], req_nm[w], integ_re[w], integ_im[w]);
        }
        integral_re = Sum(integ_re, n_omega);
        integral_im = Sum(integ_im, n_omega);
        C_re[m] = exp(-1 * integral_re) * cos(integral_im);
        C_im[m] = exp(-1 * integral_re) * sin(-1 * integral_im);
    }
    
    ss.clear();
    idstr = "";
    ss << "s" << s;
    ss << "tp" << tp ;
    idstr += ss.str();
    
    outfile.open((emptystr+"QMLSC_NEFGR_"+nameapp+idstr+".dat").c_str());
    for(i = 0 ; i < nn/2; i++) {
        omega_DA = i * d_omega_DA;
        corr1[i] = corr2[i] = 0;
        for (m=0; m<M; m++) {
            tau = m * DeltaTau;
            corr1[i] += C_re[m] * cos(omega_DA*tau) - C_im[m] * sin(omega_DA*tau);
            corr2[i] += C_re[m] * sin(omega_DA*tau) + C_im[m] * cos(omega_DA*tau);
        }
        outfile << DeltaTau * corr1[i] *2*DAcoupling*DAcoupling << endl;
    }
    outfile.close();
    outfile.clear();
    */
    
 
    //option [B]: fix omega_DA, scan tp = 0 - tp_max (main result)
    omega_DA = omega_DA_fix; //fix omega_DA

    //ss.clear();
    ss.str("");
    idstr = "";
    ss << "s" << s;
    ss << "w" << omega_DA ;
    idstr += ss.str();
    
    outfile.open((emptystr+"QMLSC_NEFGR_"+nameapp+idstr+".dat").c_str());
    
    
    outfile1.open((emptystr+"QMLSC_P_NEFGR_"+nameapp+idstr+".dat").c_str());
    

    sum=0;
    kneq=0;
    for (tp = 0; tp < tp_max; tp += Deltatp) {
        kre = kim = 0;
        M = static_cast<int> (tp/DeltaTau);//Attention!, update M for each tp
        for (m=0; m < M; m++) {//tau index
            tau = m * DeltaTau;
            integ_re[0] = 0;
            integ_im[0] = 0;
            for (w = 0; w < n_omega; w++) {
                Integrand_NE_exact(omega_nm[w], tp, tau, shift_NE[w], req_nm[w], integ_re[w], integ_im[w]);
            }
            integral_re = Sum(integ_re, n_omega);//*DeltaTau;
            integral_im = Sum(integ_im, n_omega);//*DeltaTau;
            temp_re = exp(-1 * integral_re) * cos(integral_im);
            temp_im = exp(-1 * integral_re) * sin(-1 * integral_im);
            kre += temp_re * cos(omega_DA*tau) - temp_im * sin(omega_DA*tau);
            kim += temp_re * sin(omega_DA*tau) + temp_im * cos(omega_DA*tau);
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
    

    
    
    //case [3]: C-AV approximation using discrete, fix omega_DA and scan tp
    outfile.open((emptystr+"CAV_NEFGR_"+nameapp+idstr+".dat").c_str());
    outfile1.open((emptystr+"CAV_P_NEFGR_"+nameapp+idstr+".dat").c_str());
    sum=0;
    kneq=0;
    for (tp = 0; tp < tp_max; tp += Deltatp) {
        kre = kim = 0;
        M = static_cast<int> (tp/DeltaTau);
        for (m=0; m<M; m++) {//tau index
            tau = m * DeltaTau;
            integ_re[0] = 0;
            integ_im[0] = 0;
            for (w = 0; w < n_omega; w++) {
                Integrand_NE_CAV(omega_nm[w], tp, tau, shift_NE[w], req_nm[w], integ_re[w], integ_im[w]);
            }
            integral_re = Sum(integ_re, n_omega); // *DeltaTau;
            integral_im = Sum(integ_im, n_omega); // *DeltaTau;
            temp_re = exp(-1 * integral_re) * cos(integral_im);
            temp_im = exp(-1 * integral_re) * sin(-1 * integral_im);
            kre += temp_re * cos(omega_DA*tau) - temp_im * sin(omega_DA*tau);
            kim += temp_re * sin(omega_DA*tau) + temp_im * cos(omega_DA*tau);
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
    
    //case [4]: C-D approximation using discrete, fix omega_DA and scan tp
    outfile.open((emptystr+"CD_NEFGR_"+nameapp+idstr+".dat").c_str());
    outfile1.open((emptystr+"CD_P_NEFGR_"+nameapp+idstr+".dat").c_str());
    sum=0;
    kneq=0;
    for (tp = 0; tp < tp_max; tp += Deltatp) {
        kre = kim = 0;
        M = static_cast<int> (tp/DeltaTau);
        for (m=0; m<M; m++) {//tau index
            tau = m * DeltaTau;
            integ_re[0] = 0;
            integ_im[0] = 0;
            for (w = 0; w < n_omega; w++) {
                Integrand_NE_CD(omega_nm[w], tp, tau, shift_NE[w], req_nm[w], integ_re[w], integ_im[w]);
            }
            integral_re = Sum(integ_re, n_omega); // *DeltaTau;
            integral_im = Sum(integ_im, n_omega); // *DeltaTau;
            temp_re = exp(-1 * integral_re) * cos(integral_im);
            temp_im = exp(-1 * integral_re) * sin(-1 * integral_im);
            kre += temp_re * cos(omega_DA*tau) - temp_im * sin(omega_DA*tau);
            kim += temp_re * sin(omega_DA*tau) + temp_im * cos(omega_DA*tau);
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
    
    //case [5]: W-0 inh approximation using discrete, fix omega_DA and scan tp
    outfile.open((emptystr+"W0_NEFGR_"+nameapp+idstr+".dat").c_str());
    outfile1.open((emptystr+"W0_P_NEFGR_"+nameapp+idstr+".dat").c_str());
    sum=0;
    kneq=0;
    for (tp = 0; tp < tp_max; tp += Deltatp) {
        kre = kim = 0;
        M = static_cast<int> (tp/DeltaTau);
        for (m=0; m<M; m++) {//tau index
            tau = m * DeltaTau;
            integ_re[0] = 0;
            integ_im[0] = 0;
            for (w = 0; w < n_omega; w++) {
                Integrand_NE_W0(omega_nm[w], tp, tau, shift_NE[w], req_nm[w], integ_re[w], integ_im[w]);
            }
            integral_re = Sum(integ_re, n_omega); // *DeltaTau;
            integral_im = Sum(integ_im, n_omega); // *DeltaTau;
            temp_re = exp(-1 * integral_re) * cos(integral_im);
            temp_im = exp(-1 * integral_re) * sin(-1 * integral_im);
            kre += temp_re * cos(omega_DA*tau) - temp_im * sin(omega_DA*tau);
            kim += temp_re * sin(omega_DA*tau) + temp_im * cos(omega_DA*tau);
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
    
    //case [6] Marcus approximation using discrete, fix omega_DA and scan tp
    outfile.open((emptystr+"Marcus_NEFGR_"+nameapp+idstr+".dat").c_str());
    outfile1.open((emptystr+"Marcus_P_NEFGR_"+nameapp+idstr+".dat").c_str());
    sum=0;
    kneq=0;
    for (tp = 0; tp < tp_max; tp += Deltatp) {
        kre = kim = 0;
        M = static_cast<int> (tp/DeltaTau);
        for (m=0; m<M; m++) {//tau index
            tau = m * DeltaTau;
            integ_re[0] = 0;
            integ_im[0] = 0;
            for (w = 0; w < n_omega; w++) {
                Integrand_NE_Marcus(omega_nm[w], tp, tau, shift_NE[w], req_nm[w], integ_re[w], integ_im[w]);
            }
            integral_re = Sum(integ_re, n_omega); // *DeltaTau;
            integral_im = Sum(integ_im, n_omega); // *DeltaTau;
            temp_re = exp(-1 * integral_re) * cos(integral_im);
            temp_im = exp(-1 * integral_re) * sin(-1 * integral_im);
            kre += temp_re * cos(omega_DA*tau) - temp_im * sin(omega_DA*tau);
            kim += temp_re * sin(omega_DA*tau) + temp_im * cos(omega_DA*tau);
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
    
    
    outfile2.close();
    outfile2.clear();
    
    
    cout << "--- SUMMARY --- " << endl;
    /*
    cout << "[1] fix tp, scan omega_DA" << endl;
    cout << "   fix tp = " << tp_fix << endl;
    cout << "   d_omega_DA = " << d_omega_DA << endl;
    cout << "   number of omega_DA = " << nn/2 << endl << endl;
    */
    //cout << "[2] fix omega_DA, scan tp" << endl;
    cout << "fix omega_DA = " << omega_DA_fix << endl;
    cout << "Delta tp = " << Deltatp << endl;
    cout << "number of tp = " << tp_max/Deltatp << endl << endl;
    
    cout << "normal modes n_omega = " << n_omega << endl;
    cout << "omega_max = " << omega_max << endl;
    cout << "initial shift s = " << s << endl;
    cout << "---------- END of all NEFGR in Condon case ----------" << endl;
    return 0;
}



/********* SUBROUTINE *************/


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


