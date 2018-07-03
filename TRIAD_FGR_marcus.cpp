/* This code calculates equilibrium Fermi Golden Rule rate
   in Condon case, for C-P-C_60 molecular triad
   via the Marcus level theory 
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
string Donor = "PI";   //string for state: PI, CT1, CT2
string Acceptor = "CT2";
const int D_ind = 0; //index for state: PI=0; CT1=1; CT2=2
const int A_ind = 2;
//------------------------------------------------------------------
const int INIT_TRAJ = 1041;
const int N_TRAJ = 40;// number of traj for a batch
const int BATCH = 4;// Batches of N_TRAJ trajectories for averaging
const int LEN = 512;//number of t choices for FFT
const int LEN_TRAJ_EQ = 10000; // number of PEs in EQ trajectory
const int LEN_TRAJ = 10000; // correlation function length
const int STEPS = 1; // skip time steps for TCF
const double DeltaT_fs = 1; //(fs) FFT time sampling interval
const double DAcoupling = 0.1;
const double temperature = 300;// (Kelvin) temperature
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
    int i, j, a, b;
    double *VD = new double [LEN_TRAJ_EQ];
    double *VA = new double [LEN_TRAJ_EQ];
    double *UDA = new double [LEN_TRAJ_EQ];
    
    int index(0);
    int id_traj;
    
    ss.str("");
    idstr = "";
    ss << index;
    idstr = ss.str();
    
    double DEag[3];//Energy shift a-g in a.u.
    
    cout << "---------------Energy in eV ------------------" << endl;
    
    double VD_D_avg(0);
    double VA_D_avg(0);
    double VD_A_avg(0);
    double VA_A_avg(0);
    double VD_D_total_avg(0);
    double VA_D_total_avg(0);
    double VD_A_total_avg(0);
    double VA_A_total_avg(0);
    double UDA_avg(0);
    double UDA2_avg(0);
    double UDA_total_avg(0);
    double UDA2_total_avg(0);
    double sigma2_total_avg(0);
    double k_M;
    
    double k_M_accum[BATCH];
    double UDA_avg_accum[BATCH];
    
    //compare with FFT
    double df= 1.0/LEN/DeltaT_fs; // * 10^15 s^-1
    double domega = df * 2 * pi;
    double dE = domega * hbar_eVfs; //FFT freq step in eV
    double UDA_eV;
    

    ifstream infile;
    
    cout << "---------- RESULT for " << Donor <<" -> " << Acceptor << "------------"<<endl;
    //Total averaging over BATCH N_TRAJ trajs.
    for (b = 0; b < BATCH; b++) {
        
        VD_D_avg = 0;
        VA_D_avg = 0;
        VD_A_avg = 0;
        VA_A_avg = 0;
        VD_D_total_avg  = 0; //<VD>_D
        VA_D_total_avg  = 0; //<VA>_D
        VD_A_total_avg  = 0; //<VD>_A
        VA_A_total_avg  = 0; //<VA>_A
        UDA_total_avg   = 0; //<UDA>_D
        UDA2_total_avg  = 0; //<UDA^2>_D
        sigma2_total_avg = 0;//<sigma^2>_D
        
        if (end_traj - start_traj + 1 != N_TRAJ) cout << "error start, end traj." << endl;
        
        cout << "Batch " << b << ": Input Traj # " << start_traj << "  -->  " << end_traj << ".   Total = " << N_TRAJ << endl;
        
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
            VD_D_avg = 0;
            for (i = 0; i < LEN_TRAJ_EQ; i++) {
                infile >> VD[i];
                VD[i] *= kcal2eV;  //convert to eV
                VD[i] += ECag[D_ind]; //add electronic energy correction
                VD_D_avg += VD[i];
            }
            infile.close();
            infile.clear();
            VD_D_avg /= LEN_TRAJ_EQ;
            
            
            infile.open((emptystr+"./"+Donor+"/TRIAD_RBENT_E_"+Acceptor+"_TRAJ_"+Donor+"_"+idstr+".dat").c_str());
            if (!infile.is_open()) {
                cout << "Error: cannot open input file # "<< id_traj << endl;
                return -1;
            }
            VA_D_avg = 0;
            for (i = 0; i< LEN_TRAJ_EQ; i++) {
                infile >> VA[i];
                VA[i] *= kcal2eV;  //convert to eV
                VA[i] += ECag[A_ind]; //add electronic energy correction
                VA_D_avg += VA[i];
            }
            infile.close();
            infile.clear();
            VA_D_avg /= LEN_TRAJ_EQ;
            //---------------------------------------------
            
            
            UDA_avg = 0;
            UDA2_avg = 0;
            for (i = 0; i< LEN_TRAJ_EQ; i++) {
                //calcualte energy gap and add energy shift in Ev
                UDA[i] = VD[i] - VA[i];
                UDA_avg += UDA[i];
                UDA2_avg += UDA[i] * UDA[i];
            }
            UDA_avg /= LEN_TRAJ_EQ;
            UDA2_avg /= LEN_TRAJ_EQ;
            
            UDA_total_avg += UDA_avg;
            UDA2_total_avg += UDA2_avg;
            VD_D_total_avg += VD_D_avg;
            VA_D_total_avg += VA_D_avg;
            
            
        }


        VD_D_total_avg /= N_TRAJ; //<VD>_D
        VA_D_total_avg /= N_TRAJ; //<VA>_D
        
        UDA_total_avg  /= N_TRAJ; //<UDA>_D
        UDA2_total_avg /= N_TRAJ; //<UDA^2>_D
        sigma2_total_avg = UDA2_total_avg - UDA_total_avg * UDA_total_avg; //sigma^2_D
    
        UDA_avg_accum[b] = UDA_total_avg;

    
        ss.str("");
        idstr = "";
        ss << "_"<< start_traj <<"-"<<end_traj;
        idstr = ss.str();
        
        //--- Marcus theory --------------
        
        cout << " <VD>_D (eV) = " << VD_D_total_avg << endl;
        cout << " <VA>_D (eV) = " << VA_D_total_avg << endl;
        cout << " <UDA>_D (eV) = " << UDA_total_avg << endl;
        cout << " <UDA^2>_D (eV^2) = " << UDA2_total_avg << endl;
        cout << " sigma^2_D (eV) = " << sigma2_total_avg << endl;
        
        double Er(0);//eV
        Er = sigma2_total_avg * 0.5 / (temperature * kT2eV);
        
        cout << " Er (eV) = sigma^2_D/2kT = " << Er  << endl;
        cout << " DeltaE (eV) = -Er - <UDA>_D = " << - Er - UDA_total_avg  << endl;
        
        double CGG(0);
        
        CGG = sqrt(2*pi/sigma2_total_avg) * exp(-0.5*UDA_total_avg*UDA_total_avg/sigma2_total_avg);
        
        k_M = CGG * Gamma_DA[D_ind+A_ind-1] * Gamma_DA[D_ind+A_ind-1] / hbar_eVs;
        cout << " --->   k_Marcus (s^-1) = " <<  k_M << endl;
        cout << endl;
        
        k_M_accum[b] = k_M;
        
        start_traj += N_TRAJ;
        end_traj += N_TRAJ;

    }
    
    double m, error;

    m=0; error=0;
    for (b=0; b<BATCH; b++) {
        m += k_M_accum[b];
    }
    m /= BATCH;
    for (b=0; b<BATCH; b++) {
        error += (k_M_accum[b]-m)*(k_M_accum[b]-m);
    }
    error /= BATCH;
    error = sqrt(error);
    cout << "AVG [*] --> k_Marcus (s^-1) = " << m << " (+/-) " << error << endl;
    
    m=0; error=0;
    for (b=0; b<BATCH; b++) {
        m += UDA_avg_accum[b];
    }
    m /= BATCH;
    for (b=0; b<BATCH; b++) {
        error += (UDA_avg_accum[b]-m)*(UDA_avg_accum[b]-m);
    }
    error /= BATCH;
    error = sqrt(error);
    cout << "AVG [*] --> UDA (eV) = " << m << " (+/-) " << error << endl;

    
    
    return 0;
}




