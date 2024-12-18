/********************************************************************************/
/*  StePS - STEreographically Projected cosmological Simulations                */
/*    Copyright (C) 2017-2022 Gabor Racz                                        */
/*                                                                              */
/*    This program is free software; you can redistribute it and/or modify      */
/*    it under the terms of the GNU General Public License as published by      */
/*    the Free Software Foundation; either version 2 of the License, or         */
/*    (at your option) any later version.                                       */
/*                                                                              */
/*    This program is distributed in the hope that it will be useful,           */
/*    but WITHOUT ANY WARRANTY; without even the implied warranty of            */
/*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             */
/*    GNU General Public License for more details.                              */
/********************************************************************************/

#define pi 3.14159265358979323846264338327950288419716939937510  // 50-digit precision of Pi
#define UNIT_T 47.14829951063323   //Unit time in Gy
#define UNIT_V 20.738652969925447  //Unit velocity in km/s

#ifdef GLASS_MAKING //Newtonian gravitational constant (in internal units)
    #define G -1.0  //Gravity is repulsive, if GLASS_MAKING is on.
#else
    #define G 1.0   //Normal gravity
#endif

#ifdef USE_SINGLE_PRECISION
    typedef float REAL;
#else
    typedef double REAL;
#endif

extern int IS_PERIODIC; //Periodic boundary conditions; 0: none, 1: nearest images, 2: Ewald forces
extern int COSMOLOGY; //Cosmological simulation; 0: no, 1: yes
extern int COMOVING_INTEGRATION; //Comoving integration; 0: no, 1: yes, used only when COSMOLOGY=1
extern REAL L; //Size of the simulation box
extern char IC_FILE[1024]; //Input initial conditions (IC) file
extern char OUT_DIR[1024]; //Output directory
extern char OUT_LST[1024]; //Output redshift list file. Only used when OUTPUT_TIME_VARIABLE=1.
extern int IC_FORMAT; // Format of the IC file; 0: ASCII, 1: GADGET
extern int OUTPUT_FORMAT; //Format of the output file; 0: ASCII, 2: HDF5
extern int OUTPUT_TIME_VARIABLE; // 0: time, 1: redshift
extern double MIN_REDSHIFT; //The minimal output redshift. Lower redshifts are considered to be 0.
extern int REDSHIFT_CONE; // 0: standard output files, 1: one output redshift cone file
extern int HAVE_OUT_LIST;
extern double TIME_LIMIT_IN_MINS; //Simulation wall-clock time limit in minutes
extern int H0_INDEPENDENT_UNITS; // 0: I/O in Mpc, Msol, etc. 1: I/O in Mpc/h, Msol/h, etc.
extern double *out_list; //Output redshifts
extern double *r_bin_limits; //Bin limits in D_c for redshift cone simulations
extern int out_list_size; //Number of output redshifts
extern unsigned int N_snapshot; //Number of snapshots written out
extern bool ForceError; //True, if any errors encountered during the force calculation
extern bool *IN_CONE;

extern int n_GPU; //Number of CUDA capable GPUs
//Variables for MPI
extern int numtasks, rank;
extern int N_mpi_thread; //Number of calculated forces in one MPI thread
extern int ID_MPI_min, ID_MPI_max; //max and min ID of of calculated forces in one MPI thread
extern MPI_Status Stat;
extern REAL *F_buffer; //buffer for force copy
extern int BUFFER_start_ID;

extern int e[2202][4]; //Ewald space/Ewald sphere vectors

extern REAL x4, err, errmax, ACC_PARAM; //variables used for error calculations
extern double h, h_min, h_max, t_next;  //actual stepsize, minimal and maximal stepsize, next time for output
extern double a_max, t_bigbang; //maximal scale factor; Age of Big Bang

extern double FIRST_T_OUT, H_OUT; //First output time, output frequency in Gy

extern int N; //Number of particles
extern int t; //Number of the actual timestep
extern REAL *x; //Particle coordinates
extern REAL *v; //Particle velocities
extern REAL *F; //Forces
extern REAL *M; //Particle masses
extern REAL M_tmp;
extern REAL M_min; //Minimal particle mass
extern REAL mass_in_unit_sphere; //Mass in unit sphere
extern REAL rho_part; //One particle density
extern REAL beta; //Particle radii
extern REAL particle_radii; //Particle radii; read from parameter file
extern REAL w[3]; //Parameters for smoothing in force calculation
//extern REAL SOFT_CONST[8]; //Parameters for smoothing in force calculation
extern REAL *SOFT_LENGTH; //particle softening lengths
#ifdef HAVE_HDF5
    extern int HDF5_redshiftcone_firstshell;
    extern int N_redshiftcone; //number of particles written out to the redshiftcone file
#endif
//Cosmological parameters
extern double //needed for all cosmological models
    Omega_b, Omega_lambda, Omega_dm, Omega_r, Omega_k, Omega_m,
    H0, Hubble_param, Decel_param, delta_Hubble_param;
#if COSMOPARAM==1
    extern double w0;  //Dark energy equation of state at all redshifts. (LCDM: w0=-1.0)
#elif COSMOPARAM==2
    extern double w0;  //Dark energy equation of state at z=0. (LCDM: w0=-1.0)
    extern double wa;  //Negative derivative of the dark energy equation of state. (LCDM: wa=0.0)
#elif COSMOPARAM==-1
    extern char EXPANSION_FILE[1024];  //Input file with expansion history
    extern int N_expansion_tab;        //Number of rows in the expansion history tab
    extern int expansion_index;        //Index of the current value in the expansion history
    extern double **expansion_tab;     //Expansion history tab (columns: t, a, H)
    extern int INTERPOLATION_ORDER;    //Order of the interpolation (1,2,or 3)
#endif
extern double rho_crit; //Critical density
extern double a, a_start, a_prev, a_tmp; //Scale factor, scale factor at the starting time, previous scale factor
extern double T, delta_a, Omega_m_eff; //Physical time, change of scale factor, effectve Omega_m

//Functions
//Initial timestep length calculation
double calculate_init_h();
//Functions used for the Friedmann-equation
double friedmann_solver_step(double a0, double h);
double CALCULATE_Hubble_param(double a);
void recalculate_softening();
//This function calculates the deceleration parameter
double CALCULATE_decel_param(double a);
