/* This code calculates equilibrium Fermi Golden Rule rate
   in Condon case, for C-P-C_60 molecular triad
   via the C-0 approximation 
   (c) Xiang Sun 2017
 */

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
using namespace std;

//******************************************************************
const int INIT_TRAJ = 1041;
const int N_TRAJ = 40;//19;// number of traj for a batch
const int BATCH = 4;//5; // Batches of N_TRAJ trajectories for averaging
string Donor = "PI";   //string for state: PI, CT1, CT2
string Acceptor = "CT2";
const int D_ind = 0; //index for state: PI=0; CT1=1; CT2=2
const int A_ind = 2;
const int LEN_TRAJ_EQ = 10000; // number of PEs in EQ trajectory
const int LEN_TRAJ = 10000; // correlation function length
const int LEN = 512; //512;//number of t choices for FFT
const int STEPS = 1; // skip time steps for TCF
const int skip = 20; // skip time steps for UDA
const double DeltaT_fs = 1; //(fs) FFT time sampling interval
const double temperature = 300;// (Kelvin) temperature
const int MAXBIN = 200;
double UDA_min = -2.5; // eV
double UDA_max = 1.5; // eV
double ECag[3] = {0.830284 , -2.18684,  -1.4569};//Energy correction in eV for BENT triad
double Gamma_DA[3] = {2.4e-2 , 4.5e-5,  8.6e-5};//BENT triad, diabatic coupling in eV
//double ECag[3] = {1.582 , -1.140,  -1.186};//Energy correction in eV for LINEAR triad
//double Gamma_DA[3] = {9.0e-3 , 2.0e-4,  1.0e-3};//LINEAR triad, diabatic coupling in eV
//******************************************************************


const double fs2au = 41.341105; //1 fs = 41.34 a.u.
const double kT2au = 3.168429139e-6;// kT(1 Kelvin) = 3.2e-6 a.u.
const double kcal2au = 0.00159362; // 1 kcal/mol = 0.00159362 hartree (a.u.)
const double eV2au = 0.0367502;// 1 eV = 0.0367502 hartree (a.u.)
const double kcal2eV = 0.0433634; // 1 kcal/mol = 0.0433634 eV
const double kT2eV = 0.0000861705;// kT(1 Kelvin) = 0.0000861705 eV

double DeltaT = DeltaT_fs * fs2au; // in (a.u.)
double kT = temperature * kT2au; //in (a.u.)
double beta = 1.0 / kT; //in (a.u.)
const double pi=3.14159265358979324;
const double RT_2PI= sqrt(2*pi);
const double hbar = 1;
const double hbar_eVs = 6.582120e-16; //(eV*s)
const double hbar_eVfs = 6.582120e-1;// (eV*fs)


void FFT(int dir, int m, double *x, double *y); //Fast Fourier Transform, 2^m data


int main (int argc, char *argv[]) {
    
    int start_traj = INIT_TRAJ;
    int end_traj = INIT_TRAJ + N_TRAJ -1;
    
    stringstream ss;
    string emptystr("");
    string filename;
    string idstr("");
    string nameapp("");

    ofstream outfile;
    ss.str("");
    nameapp = "";
    ss << Donor << "->" << Acceptor;
    nameapp = ss.str();
    
    double t;
    int i, j, k, a, b;
    double *VD = new double [LEN_TRAJ_EQ];
    double *VA = new double [LEN_TRAJ_EQ];
    double *UDA = new double [LEN_TRAJ_EQ*N_TRAJ];
    double uda;
    double UDA_avg(0);

    int index(0);
    int id_traj;
    
    ss.str("");
    idstr = "";
    ss << index;
    idstr = ss.str();
    
    //cout << "---------------Energy in eV ------------------" << endl;
    cout << "---------- " << Donor <<" -> " << Acceptor << "------------"<<endl;
    
    ifstream infile;
    
    double *Ct_re = new double [LEN_TRAJ];
    double *Ct_im = new double [LEN_TRAJ];
    double integral_re, integral_im;
    int count(0);
    double k_C0(0);
    double k_C0_avg(0);
    double IUDA(0);
    
    double k_C0_FFT[BATCH];
    
    //---------- FFT variables ----------------------------------------------
    int mm(0), nn(1); // nn = 2^mm is number of (complex) data to FFT
    double T0 = - DeltaT_fs * (LEN * 0.5);//time origin in fs
    double shift = T0 / DeltaT_fs; //time origin shift index
    while (nn < LEN ) {
        mm++;
        nn *= 2;
    } //nn is the first 2^m that larger LEN
    
    double *corr1 = new double [nn];
    double *corr2 = new double [nn];
    
    double df= 1.0/LEN/DeltaT_fs; //10^15 s^-1
    double domega = df * 2 * pi;
    
    double *corr1_orig = new double [nn]; //shifted origin to T0
    double *corr2_orig = new double [nn];
    //------------------------------------------------------------------------
    
    
    
    for (b = 0; b < BATCH; b++) { //Total averaging over BATCH N_TRAJ trajs.
        
        if (end_traj - start_traj + 1 != N_TRAJ) cout << "error start, end traj." << endl;

        cout << "Batch " << b << ": Input Traj # " << start_traj << "  -->  " << end_traj << ".   Total = " << N_TRAJ << endl;
        
        UDA_avg = 0;
        for (id_traj = start_traj; id_traj <= end_traj; id_traj++)  {
            ss.str("");
            idstr = "";
            ss << id_traj;
            idstr = ss.str();
            
            //----------------- Traj on donor PES --------------------
            infile.open((emptystr+"./"+Donor+"/TRIAD_RBENT_E_"+Donor+"_TRAJ_"+Donor+"_"+idstr+".dat").c_str());
            if (!infile.is_open()) {
                cout << "Error: cannot open input file # " << id_traj << endl;
                return -1;
            }
            for (i = 0; i < LEN_TRAJ_EQ; i++) {
                infile >> VD[i];
                VD[i] *= kcal2eV;  //convert to eV
                VD[i] += ECag[D_ind]; //add electronic energy correction
            }
            infile.close();
            infile.clear();
            
            infile.open((emptystr+"./"+Donor+"/TRIAD_RBENT_E_"+Acceptor+"_TRAJ_"+Donor+"_"+idstr+".dat").c_str());
            if (!infile.is_open()) {
                cout << "Error: cannot open input file # "<< id_traj << endl;
                return -1;
            }
            for (i = 0; i< LEN_TRAJ_EQ; i++) {
                infile >> VA[i];
                VA[i] *= kcal2eV;  //convert to eV
                VA[i] += ECag[A_ind]; //add electronic energy correction
            }
            infile.close();
            infile.clear();

            for (i = 0; i < LEN_TRAJ_EQ; i++) {
                //calcualte energy gap and add energy shift in Ev
                uda = VD[i] - VA[i];
                UDA[i + (id_traj-start_traj) * LEN_TRAJ_EQ] = uda;
                UDA_avg += uda;
            }
        }
        //calculate average value of UDA
        UDA_avg /= LEN_TRAJ_EQ*N_TRAJ;

        
        ss.str("");
        idstr = "";
        ss << "_"<< start_traj <<"-"<<end_traj;
        idstr = ss.str();
        
        
        //C-0 approx using FFT of time-domain UDA.

        for (i = 0; i < nn; i++) corr1[i] = corr2[i] = 0;
        count = 0;
        
        for (j = 0; j< LEN_TRAJ_EQ * N_TRAJ; j++) {
            for (k = 0; k < LEN; k++) {
                t = T0 + DeltaT_fs * k; // in fs
                integral_re = (UDA[j]-UDA_avg) * t / hbar_eVfs;
                
                corr1[k] += cos(integral_re);
                corr2[k] += sin(integral_re);
            }
            count++;
        }
        
        for (k = 0; k < LEN; k++) {
            corr1[k] /=count;
            corr2[k] /=count;
        }
        
        
        corr1[LEN/2] = 1;
        corr2[LEN/2] = 0;
        
        
        FFT(-1, mm, corr1, corr2);
        
        for(i = 0; i < nn; i++) { //shift time origin
            corr1_orig[i] = corr1[i] * cos(2*pi*i*shift/nn) - corr2[i] * sin(-2*pi*i*shift/nn);
            corr2_orig[i] = corr2[i] * cos(2*pi*i*shift/nn) + corr1[i] * sin(-2*pi*i*shift/nn);
        }
        
        double prefactor = LEN * DeltaT_fs * Gamma_DA[D_ind+A_ind-1] * Gamma_DA[D_ind+A_ind-1] / hbar_eVfs / hbar_eVs;
        
        double FFTav;

        if (UDA_avg > 0) {
            k = static_cast<int> (UDA_avg/domega/hbar_eVfs+0.5);
            if (k >= nn/2) cout << "Error in FFT: <UDA> out of range."<< endl;
        }
        else if (UDA_avg < 0) {
            k = nn + static_cast<int> (UDA_avg/domega/hbar_eVfs+0.5);
            if (k < nn/2) cout << "Error in FFT: -<UDA> out of range."<< endl;
        }
        
        FFTav = corr1_orig[k];
        k_C0 = FFTav * prefactor;
        cout << "    k_C0 (s^-1) using FFT = " << k_C0  << endl;
        k_C0_FFT[b] = k_C0;
        cout << "    <UDA> = " << UDA_avg << " eV   -> index = " << k << endl;
        
        //next iteration of averaging.
        start_traj += N_TRAJ;
        end_traj += N_TRAJ;
        cout << "------------------------------" << endl;
    }
        
        
    cout << "    LEN for FFT = " << LEN << endl;
    cout << "    hbar * d omega   = " << domega * hbar_eVfs << endl;
    cout << "    hbar * omega_max   = " << LEN/2 * domega * hbar_eVfs << endl;
    
    
    //Error bar analysis
    double m, error;
    
    m=0; error=0;
    for (b=0; b<BATCH; b++) {
        m += k_C0_FFT[b];
    }
    m /= BATCH;
    for (b=0; b<BATCH; b++) {
        error += (k_C0_FFT[b]-m)*(k_C0_FFT[b]-m);
    }
    error /= BATCH;
    error = sqrt(error);
    cout << "AVG [*] --> k_C0 (s^-1) = " << m << " (+/-) " << error << endl;
    
    
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
        
        n = 1;
        for (i=0;i<m;i++)
            n *= 2;
        
        i2 = n >> 1;
        j = 0;
        for (i=0;i<n-1;i++) {
            if (i < j) {
                tx = x[i];
                ty = y[i];
                x[i] = x[j];
                y[i] = y[j];
                x[j] = tx;
                y[j] = ty;
            }
            k = i2;
            while (k <= j) {
                j -= k;
                k >>= 1;
            }
            j += k;
        }
        
        c1 = -1.0;
        c2 = 0.0;
        l2 = 1;
        for (l=0;l<m;l++) {
            l1 = l2;
            l2 <<= 1;
            u1 = 1.0;
            u2 = 0.0;
            for (j=0;j<l1;j++) {
                for (i=j;i<n;i+=l2) {

                    i1 = i + l1;
                    t1 = u1 * x[i1] - u2 * y[i1];
                    t2 = u1 * y[i1] + u2 * x[i1];
                    x[i1] = x[i] - t1;
                    y[i1] = y[i] - t2;
                    x[i] += t1;
                    y[i] += t2;
                }
                
                z =  u1 * c1 - u2 * c2;
                u2 = u1 * c2 + u2 * c1;
                u1 = z;
            }
            
            c2 = sqrt((1.0 - c1) / 2.0);
            if (dir == 1)
                c2 = -c2;
            c1 = sqrt((1.0 + c1) / 2.0);
        }
        
        
        if (dir == -1) {
            for (i=0;i<n;i++) {
                x[i] /= n;
                y[i] /= n;
            }
        }
        
        return;
    }




