/* This code calculates numerical equilibrium Fermi Golden Rule rate
   in Condon case, Brownian oscillator model using EXACT NORMAL MODES
   compare with linearized semiclassical methods using Monte Carlo sampling
   To compile: g++ -o EFGR_num EFGR_num.cpp -llapack -lrefblas -lgfortran
   (c) Xiang Sun 2015
 */

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
#include "r_1279.h"  // r1279 random number generator
using namespace std;

// *********** change parameters *********
const int MCN = 5000;//Monte Carlo sampling rate
double Omega = 0.2; //primary mode freq
double beta = 1;//0.2;//1;//5;
double eta = 1; //0.2;//1;//5;
const int n_omega = 1000;
const double omega_max = 15;
const int LEN = 512; //512;//number of t choices or 1024 with DeltaT=0.3
const double DeltaT = 0.2;//0.2;//0.3;//FFT time sampling interval
double DT=0.002; //MD time step
const double DAcoupling = 0.1;
// *********** **************** *********

const int INIT_omega = 0; //first omega index, =0 for normal modes, =1 for Jeff
const double y_0 = 1.0; //shift of primary mode
const double d_omega = omega_max / n_omega;//SD evenly sampling rate
const double d_omega_eff = omega_max / n_omega; //for Jeffective SD
const double omega_c = 1; // the cutoff frequency for ohmic
const double T0 = -DeltaT*(LEN*0.5);//-DeltaT*LEN/2+DeltaT/2;
const double pi =3.14159265358979324;
const double RT_2PI = sqrt(2*pi);
const double hbar = 1;
double DTSQ2 = DT * DT * 0.5;
double DT2 = DT/2.0;
double ABSDT = abs(DT);
const int N = n_omega; //number of degrees of freedom

void FFT(int dir, int m, double *x, double *y); //Fast Fourier Transform, 2^m data
void DFT(int dir, int m, double *x, double *y); //Discrete Fourier Transform
double J_omega_ohmic(double omega, double eta);//bath Ohmic SD
double J_omega_ohmic_eff(double omega, double eta); //effective SD for Ohmic bath
double S_omega_ohmic(double omega, double eta); //ohmic with decay spectral density
double S_omega_drude(double omega, double eta);//Drude spectral density
double S_omega_gaussian(double omega, double eta, double sigma, double omega_op);//gaussian spectral density
void Integrand_LSC(double omega, double t, double &re, double &im);
void Integrand_LSC_inh(double omega, double t, double &re, double &im);
void Integrand_CL_avg(double omega, double t, double &re, double &im);
void Integrand_CL_donor(double omega, double t, double &re, double &im);
void Integrand_2cumu(double omega, double t, double &re, double &im);
void Integrand_2cumu_inh(double omega, double t, double &re, double &im);
double Integrate(double *data, int n, double dx);
double Integrate_from(double *data, int sp, int n, double dx);
double Sum(double *data, int n);
double** Create_matrix(int row, int col);
double GAUSS(long *seed);
void MOVEA(double R[], double V[], double F[]);
void MOVEB(double V[], double F[]);
void force_avg(double R[], double F[], double omega[], double req[]);
double DU(double R[], double omega[], double req[]);
double DUi(double R[], double omega[], double req[], int i);
int Job_finished(int &jobdone, int count, int total, int startTime);

extern "C" {
    void dsyev_(const char &JOBZ, const char &UPLO, const int &N, double *A,
                const int &LDA, double *W, double *WORK, const int &LWORK,
                int &INFO);
}

int main (int argc, char *argv[]) {
    
    int id;
    stringstream ss;
    string emptystr("");
    string filename;
    string idstr("");
    string nameapp("");
    
    //cout << "# of argument: " << argc-1 << endl;
    if (argc > 1) {
        ss << argv[1];
        idstr += ss.str();
        ss >> id;
        ss.clear();
    }
    
    ss.str("");
    nameapp = "";
    ss << "Omega" << Omega << "_b" << beta << "e" << eta << "_" << MCN << "_";
    nameapp = ss.str();

    cout << ">>> Start Job id # " << id << " of num EFGR in Condon case." << endl;
    
    int mm(0), nn(1); // nn = 2^mm is number of (complex) data to FFT
	
	while (nn < LEN ) {
		mm++;
		nn *= 2;
	} //nn is the first 2^m that larger LEN
	
    ofstream outfile;
    int i, j, a, b;
    int w; //count of omega
    double omega;
    double t;
	double *corr1 = new double [nn];
	double *corr2 = new double [nn];
    double *corr1_orig = new double [nn]; //shifted origin to T0
    double *corr2_orig = new double [nn];
    double shift = T0 / DeltaT;
    double N = nn;
    
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
    double S_nm[n_omega];//Huang-Rhys factor for normal modes
    double **D_matrix;// the Hessian matrix
    D_matrix = Create_matrix(n_omega, n_omega);
    double c_bath[n_omega]; //secondary bath mode min shift coefficients
    double gamma_array[n_omega];//linear coupling coefficients
    double shift_NE[n_omega]; //the s_j shifting for noneq initial sampling
    double d_omega_DA = 2 * pi / LEN / DeltaT; //omega_DA griding size
    double Er = 0;
    double a_parameter = 0;
    int jobdone(0);
    int startTime;
    startTime = time(NULL);
    long seed;
    seed = seedgen();	/* have seedgen compute a random seed */
    setr1279(seed);		/* seed the genertor */
    r1279(); //[0,1] random number
    
    cout << "---------- Eq FGR in Condon case ----------" << endl;
    
    //secondary bath mode min shift coefficients (for EXACT discrete normal mode analysis)
    for (w = 1; w < n_omega; w++) {
        //Ohmic bath SD
        c_bath[w] = sqrt( 2.0 / pi * J_omega_ohmic(w*d_omega, eta) * d_omega * d_omega * w);
        //Er_bath += 2.0 * c_bath[w] * c_bath[w] / (w*d_omega * w*d_omega);
    }

    //********** BEGIN of Normal mode analysis ***********
    for (i = 0; i < n_omega; i++)
        for (j = 0; j <n_omega; j++) D_matrix[i][j] = 0;
    D_matrix[0][0] = Omega*Omega;
    for (w =1 ; w < n_omega ; w++) {
        D_matrix[0][0] += pow(c_bath[w]/(w*d_omega) ,2);
        D_matrix[0][w] = D_matrix[w][0] = c_bath[w];
        D_matrix[w][w] = pow(w*d_omega ,2);
    }
    
    for (i = 0; i < dim; i++) {
        for (j = 0; j < dim; j++) {
            matrix[j][i] = D_matrix[i][j]; //switch i j to match with Fortran array memory index
        }
    }
    
    //diagonalize matrix, the eigenvectors transpose is in result matrix => TT_ns.
    dsyev_('V', 'L', col, matrix[0], col, eig_val, work, lwork, info); //diagonalize matrix
    if (info != 0) cout << "Lapack failed. " << endl;
    
    for (i = 0; i < dim; i++) omega_nm[i] = sqrt(eig_val[i]);//normal mode freqs
    
    for (i=0; i < dim; i++)
        for (j=0; j < dim; j++) TT_ns[i][j] = matrix[i][j];//transformation matrix
    
    // the coefficients of linear electronic coupling in normal modes (gamma[j]=TT_ns[j][0]*gamma_y), here gamma_y=1
    //for (i=0; i<n_omega; i++) gamma_array[i] = TT_ns[i][0];
    //for (i=0; i<n_omega; i++) shift_NE[i] = s * gamma_array[i];
    
    //req of normal modes (acceptor's potential energy min shift)
    for (i = 0; i < n_omega; i++) {
        req_nm[i] = 1 * TT_ns[i][0];
        for (a = 1; a < n_omega; a++) req_nm[i] -= TT_ns[i][a] * c_bath[a] / (a*d_omega * a*d_omega);
    }
    
    for (i = 0; i < n_omega; i++) {
        //tilde c_j coupling strength normal mode
        c_nm[i] = req_nm[i] * omega_nm[i] * omega_nm[i];
        req_nm[i] *= 2.0 * y_0;
        //discrete Huang-Rhys factor
        S_nm[i] = omega_nm[i] * req_nm[i] * req_nm[i] * 0.5;
    }

    //******** END of Normal mode analysis **************
    

    //calculate exact reorganization energy Er for Marcus theory
    Er = a_parameter = 0;
    for (i = 0; i < n_omega; i++) Er += 2.0 * c_nm[i] * c_nm[i] / (omega_nm[i] * omega_nm[i]);
    //for (i = 0; i < n_omega; i++) Er += 0.5 * omega_nm[i] * omega_nm[i] * req_nm[i] * req_nm[i]; //S_nm[i] * omega_nm[i];

    for (i = 0; i < n_omega; i++) a_parameter += 0.5 * S_nm[i] * omega_nm[i] * omega_nm[i] /tanh(beta*hbar* omega_nm[i] *0.5);
    cout << "Er = " << Er << endl;
    cout << "a_parameter = " << a_parameter << endl;
    
    
    // numerical LSC approximation of EFGR using Monte Carlo
    int MDlen;
    int tau;
    int LENMD = static_cast<int>(LEN*DeltaT/DT); //number of steps for MD
    int NMD;
    double mc_re[LEN];
    double mc_im[LEN];
    double R0[n_omega];//wigner initial sampling
    double V0[n_omega];
    double R[n_omega]; //avg dynamics w/ wigner sampling (LSC)
    double V[n_omega];
    double F[n_omega];
    double *du_accum = new double [LENMD];
    double sigma_x[n_omega];//standard deviation of position
    double sigma_p[n_omega];//standard deviation of velocity
    double integral_du[LEN];
    double t_array[LEN];  // FFT time variable t = T0, T0+DeltaT...
    
    //set standard deviation of initial sampling
    req_nm[0] = sigma_x[0] = sigma_p[0] = 0;
    for (w = INIT_omega; w < n_omega; w++) {
        sigma_x[w] = sqrt(hbar/(2*omega_nm[w]*tanh(0.5*beta*hbar*omega_nm[w])));
        sigma_p[w] = sigma_x[w] * omega_nm[w];
    }
    
    for (i = 0; i < LEN; i++) {//prepare FFT time array
        t_array[i] = T0 + DeltaT * i;
        mc_re[i] = mc_im[i] = 0;
    }
    
    int NMD_forward = static_cast<int>(abs(t_array[LEN-1]/DT));
    int NMD_backward = static_cast<int>(abs(t_array[0]/DT));
    if(NMD_forward > NMD_backward) NMD = NMD_forward; //get max of NMD_forward and NMD_backward
    else NMD = NMD_backward;
    
    
    //Begin Monte Carlo importance sampling
    
    for (j = 0; j < MCN; j++) { //Monte Carlo phase-space integration (R,P)
        for (w = INIT_omega ; w < n_omega; w++) {
            R0[w] = R[w] = GAUSS(&seed) * sigma_x[w];//Wigner initial conf sampling
            V0[w] = V[w] = GAUSS(&seed) * sigma_p[w];//Wigner initial momentum sampling
        }
        
        // --->>> Forward MD propagation
        if (t_array[LEN-1] < 0) DT = - ABSDT; //for MD propagation direction
        else DT = ABSDT;
        DT2= 0.5 * DT;
        //NMD = NMD_forward; NMD now is max of NMD_forward and NMD_backward
        
        //dynamics on average surface + wigner sampling
        force_avg(R, F, omega_nm, req_nm);
        for (tau = 0 ; tau < NMD; tau++) {
            //record DU every DT: DU is sum of all frequencies
            du_accum[tau] = DU(R, omega_nm, req_nm);
            MOVEA(R, V, F);
            force_avg(R, F, omega_nm, req_nm);
            MOVEB(V, F);
        }
        
        for (i = 0; i < LEN; i++) {
            if (t_array[i] >= 0) {
                MDlen = static_cast<int>(abs(t_array[i] / DT));
                integral_du[i] = Integrate(du_accum, MDlen, DT); //notice sign of DT
                mc_re[i] += cos(integral_du[i]/hbar);
                mc_im[i] += sin(integral_du[i]/hbar);
            }
        }
        
        // ---<<< Backward MD propagation
        for (w = INIT_omega; w < n_omega; w++) {
            R[w] = R0[w]; //restore initial condition
            V[w] = V0[w];
        }
        if (t_array[0] < 0) DT = - ABSDT; //for MD propagation direction
        else DT = ABSDT;
        DT2= 0.5 * DT;
        //NMD = NMD_backward; NMD now is max of NMD_forward and NMD_backward
        
        //dynamics on average surface + wigner sampling
        force_avg(R, F, omega_nm, req_nm);
        for (tau = 0 ; tau< NMD; tau++) {
            //record DU every DT: DU is sum of all frequencies
            du_accum[tau] = DU(R, omega_nm, req_nm);
            MOVEA(R, V, F);
            force_avg(R, F, omega_nm, req_nm);
            MOVEB(V, F);
        }
        
        for (i = 0; i < LEN; i++) {
            if (t_array[i] < 0) {
                MDlen = static_cast<int>(abs(t_array[i]/ DT)); // MDlen should >= 0
                integral_du[i] = Integrate(du_accum, MDlen, DT); //notice sign of DT
                mc_re[i] += cos(integral_du[i]/hbar);
                mc_im[i] += sin(integral_du[i]/hbar);
            }
        }
        Job_finished(jobdone, j, MCN, startTime);
    }
    
    //average Monte Carlo LSC (dynamics on average surface + wigner sampling)
    for (i = 0; i < LEN; i++) { //Monte Carlo averaging
        mc_re[i] /= MCN;
        mc_im[i] /= MCN;
        corr1[i] = mc_re[i]; //  k(t) re
        corr2[i] = mc_im[i]; //  k(t) im
    }
    
    FFT(-1, mm, corr1, corr2);//notice its inverse FT
    
    for(i = 0; i < nn; i++) { //shift time origin
        corr1_orig[i] = corr1[i] * cos(2*pi*i*shift/nn) - corr2[i] * sin(-2*pi*i*shift/nn);
        corr2_orig[i] = corr2[i] * cos(2*pi*i*shift/nn) + corr1[i] * sin(-2*pi*i*shift/nn);
    }
    
    
    outfile.open((emptystr+"num_LSC_EFGR_nm_"+nameapp+idstr+".dat").c_str());
    for (i=0; i<nn/2; i++) outfile << corr1_orig[i]*LEN*DeltaT*DAcoupling*DAcoupling << endl;
    outfile.close();
    outfile.clear();
    
    
    
    /*
    //Case [1]: Equilibrium exact QM / LSC in Condon case using discreitzed J(\omega)
    for (i = 0; i < nn; i++) corr1[i] = corr2[i] = 0; //zero padding
    for (i = 0; i < LEN; i++) {
        t = T0 + DeltaT * i;
        integ_re[0] = 0;
        integ_im[0] = 0;
        for (w = 0; w < n_omega; w++) {
            Integrand_LSC(omega_nm[w], t, integ_re[w], integ_im[w]);
            integ_re[w] *= S_nm[w];
            integ_im[w] *= S_nm[w];
        }
        integral_re = Sum(integ_re, n_omega);
        integral_im = Sum(integ_im, n_omega);
        corr1[i] = exp(-1 * integral_re) * cos(integral_im);
        corr2[i] = -1 * exp(-1 * integral_re) * sin(integral_im);
     }

     FFT(-1, mm, corr1, corr2);//notice its inverse FT
     
     for(i=0; i<nn; i++) { //shift time origin
         corr1_orig[i] = corr1[i] * cos(2*pi*i*shift/nn) - corr2[i] * sin(-2*pi*i*shift/nn);
         corr2_orig[i] = corr2[i] * cos(2*pi*i*shift/nn) + corr1[i] * sin(-2*pi*i*shift/nn);
     }
     
     outfile.open((emptystr + "QMLSC_EFGR_nm_" + nameapp + ".dat").c_str());
     for (i=0; i<nn/2; i++) outfile << corr1_orig[i]*LEN*DeltaT*DAcoupling*DAcoupling << endl;
     outfile.close();
     outfile.clear();
    */

    
    
    
    //-------------- Summary ----------------
    cout << endl;
    cout << "----- Parameters ------- " << endl;
    cout << "Omega (primary) = " << Omega << endl;
    cout << "n_omega = " << n_omega << endl;
    cout << "omega_max = " << omega_max << endl;
    cout << "beta = " << beta << endl;
    cout << "eta  = " << eta << endl;
    cout << "NMC = " << MCN << endl;
    cout << "LEN = " << LEN << endl;
    cout << "DeltaT = " << DeltaT << endl;
    cout << "--------- END of EFGR in Condon case --------" << endl;
 

    return 0;
}



/********* SUBROUTINE *************/


//spectral densities

double S_omega_ohmic(double omega, double etaa) {
    return etaa * omega * exp(-1 * omega  / omega_c);
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
    return etaa * omega * pow(Omega,4) / ( pow(Omega*Omega - omega*omega, 2) + etaa*etaa*omega*omega);
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


double** Create_matrix(int row, int col) {
    double **matrix = new double* [col];
    matrix[0] = new double [col*row];
    for (int i = 1; i < col; ++i)
        matrix[i] = matrix[i-1] + row;
    return matrix;
}

// generate Normal distribution of random variable(rannum) from a uniform distribution ran()
double GAUSS(long *seed) {
    double A1 = 3.949846138;
    double A3 = 0.252408784;
    double A5 = 0.076542912;
    double A7 = 0.008355968;
    double A9 = 0.029899776;
    double SUM, R, R2, random, rannum;
    SUM = 0.0;
    
    for (int i=1; i<=12; i++) {
        //random = ran2(seed);
        random = r1279();
        SUM += random;
    }
    R = (SUM - 6.0)/4.0;
    R2 = R*R;
    rannum = ((((A9*R2+A7)*R2+A5)*R2+A3)*R2+A1)*R;
    return (rannum);
}

// SUROUTINE TO PERFORM VELOCITY VERLET ALGORITHM
void MOVEA(double R[], double V[], double F[]) {
    double xx;
    for (int i= INIT_omega ; i<N; i++) {
        xx = R[i] + DT*V[i] + DTSQ2*F[i];
        //pbc(xx, yy, zz);
        R[i] = xx;
        V[i] += DT2*F[i];
    }
    return;
}


void MOVEB(double V[], double F[]) {
    //always call MOVEB after call force** to update force F[]
    for (int i = INIT_omega ; i<N; i++) {
        V[i] += DT2*F[i];
    }
    return;
}

void force_avg(double R[], double F[], double omega[], double req[]) {
    //avg harmonic oscillator potential
    for (int i = INIT_omega; i < N; i++) {
        F[i] = - omega[i] * omega[i] * (R[i]- req[i] * 0.5);
    }
    return;
}

double DU(double R[], double omega[], double req[]) {
    double du=0;
    for (int i = INIT_omega; i < N; i++) du += req[i]*omega[i]*omega[i]*R[i]-0.5*req[i]*req[i]*omega[i]*omega[i];
    return du;
}

double DUi(double R[], double omega[], double req[], int i) {
    double du=0;
    du = req[i]*omega[i]*omega[i]*R[i]-0.5*req[i]*req[i]*omega[i]*omega[i];
    return du;
}

int Job_finished(int &jobdone, int count, int total, int startTime) {
    int tenpercent;
    int currentTime;
    tenpercent = static_cast<int> (10 * static_cast<double> (count)/ static_cast<double> (total) );
    if ( tenpercent > jobdone ) {
        jobdone = tenpercent;
        currentTime = time(NULL);
        cout << "Job finished "<< jobdone <<"0 %. Time elapsed " << currentTime - startTime << " sec." << endl;
    }
    return tenpercent;
}


