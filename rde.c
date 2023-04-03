/*********************************************************************************/
/*                                                                               */
/*  Animation of reaction-diffusion equation in a planar domain                  */
/*                                                                               */
/*  N. Berglund, January 2022                                                    */
/*                                                                               */
/*  Feel free to reuse, but if doing so it would be nice to drop a               */
/*  line to nils.berglund@univ-orleans.fr - Thanks!                              */
/*                                                                               */
/*  compile with                                                                 */
/*  gcc -o rde rde.c                                                             */
/* -L/usr/X11R6/lib -ltiff -lm -lGL -lGLU -lX11 -lXmu -lglut -O3 -fopenmp        */
/*                                                                               */
/*  OMP acceleration may be more effective after executing                       */
/*  export OMP_NUM_THREADS=2 in the shell before running the program             */
/*                                                                               */
/*  To make a video, set MOVIE to 1 and create subfolder tif_bz                  */
/*  It may be possible to increase parameter PAUSE                               */
/*                                                                               */
/*  create movie using                                                           */
/*  ffmpeg -i wave.%05d.tif -vcodec libx264 wave.mp4                             */
/*                                                                               */
/*********************************************************************************/

/*********************************************************************************/
/*                                                                               */
/* NB: The algorithm used to simulate the wave equation is highly paralellizable */
/* One could make it much faster by using a GPU                                  */
/*                                                                               */
/*********************************************************************************/

#include <math.h>
#include <string.h>
#include <GL/glut.h>
#include <GL/glu.h>
#include <unistd.h>
#include <sys/types.h>
#include <tiffio.h>     /* Sam Leffler's libtiff library. */
#include <omp.h>
#include <time.h>

#define MOVIE 1         /* set to 1 to generate movie */
#define DOUBLE_MOVIE 1  /* set to 1 to produce movies for wave height and energy simultaneously */
#define SAVE_MEMORY 1   /* set to 1 to save memory when writing tiff images */
#define NO_EXTRA_BUFFER_SWAP 1    /* some OS require one less buffer swap when recording images */

#define VARIABLE_IOR 1      /* set to 1 for a variable index of refraction */
#define IOR 5               /* choice of index of refraction, see list in global_pdes.c */
#define MANDEL_IOR_SCALE -0.05   /* parameter controlling dependence of IoR on Mandelbrot escape speed */


/* General geometrical parameters */

#define WINWIDTH 	1920  /* window width */
#define WINHEIGHT 	1150  /* window height */
#define NX 3840          /* number of grid points on x axis */
#define NY 2300          /* number of grid points on y axis */

#define XMIN -2.0
#define XMAX 2.0	/* x interval */
#define YMIN -1.197916667
#define YMAX 1.197916667	/* y interval for 9/16 aspect ratio */

#define HIGHRES 1       /* set to 1 if resolution of grid is double that of displayed image */

#define JULIA_SCALE 1.0 /* scaling for Julia sets */

/* Choice of the billiard table */

#define B_DOMAIN 999        /* choice of domain shape, see list in global_pdes.c */

#define CIRCLE_PATTERN 1   /* pattern of circles or polygons, see list in global_pdes.c */

#define COMPARISON 0        /* set to 1 to compare two different patterns (beta) */
#define B_DOMAIN_B 20       /* second domain shape, for comparisons */
#define CIRCLE_PATTERN_B 0  /* second pattern of circles or polygons */

#define P_PERCOL 0.25       /* probability of having a circle in C_RAND_PERCOL arrangement */
#define NPOISSON 300        /* number of points for Poisson C_RAND_POISSON arrangement */
#define RANDOM_POLY_ANGLE 1 /* set to 1 to randomize angle of polygons */

#define LAMBDA 0.5	    /* parameter controlling the dimensions of domain */
#define MU 0.5              /* parameter controlling the dimensions of domain */
#define NPOLY 6             /* number of sides of polygon */
#define APOLY 0.0           /* angle by which to turn polygon, in units of Pi/2 */ 
#define MDEPTH 6            /* depth of computation of Menger gasket */
#define MRATIO 3            /* ratio defining Menger gasket */
#define MANDELLEVEL 1000    /* iteration level for Mandelbrot set */
#define MANDELLIMIT 10.0    /* limit value for approximation of Mandelbrot set */
#define FOCI 1              /* set to 1 to draw focal points of ellipse */
#define NGRIDX 14           /* number of grid point for grid of disks */
#define NGRIDY 8            /* number of grid point for grid of disks */

#define X_SHOOTER -0.2
#define Y_SHOOTER -0.6
#define X_TARGET 0.4
#define Y_TARGET 0.7        /* shooter and target positions in laser fight */

#define ISO_XSHIFT_LEFT -2.9
#define ISO_XSHIFT_RIGHT 1.4
#define ISO_YSHIFT_LEFT -0.15
#define ISO_YSHIFT_RIGHT -0.15 
#define ISO_SCALE 0.5           /* coordinates for isospectral billiards */

/* You can add more billiard tables by adapting the functions */
/* xy_in_billiard and draw_billiard below */

/* Physical parameters of wave equation */

#define TWOSPEEDS 0          /* set to 1 to replace hardcore boundary by medium with different speed */
#define OSCILLATE_LEFT 0     /* set to 1 to add oscilating boundary condition on the left */
#define OSCILLATE_TOPBOT 0   /* set to 1 to enforce a planar wave on top and bottom boundary */
#define OSCILLATION_SCHEDULE 1  /* oscillation schedule, see list in global_pdes.c */

#define OMEGA 0.0005       /* frequency of periodic excitation */
#define AMPLITUDE 0.8      /* amplitude of periodic excitation */ 
#define ACHIRP 0.25        /* acceleration coefficient in chirp */
#define DAMPING 0.0        /* damping of periodic excitation */
#define COURANT 0.05       /* Courant number */
#define COURANTB 0.0       /* Courant number in medium B */
#define GAMMA 0.0          /* damping factor in wave equation */
#define GAMMAB 0.0         /* damping factor in wave equation */
#define GAMMA_SIDES 1.0e-4      /* damping factor on boundary */
#define GAMMA_TOPBOT 1.0e-7     /* damping factor on boundary */
#define KAPPA 0.0           /* "elasticity" term enforcing oscillations */
#define KAPPA_SIDES 5.0e-4  /* "elasticity" term on absorbing boundary */
#define KAPPA_TOPBOT 0.0    /* "elasticity" term on absorbing boundary */
/* The Courant number is given by c*DT/DX, where DT is the time step and DX the lattice spacing */
/* The physical damping coefficient is given by GAMMA/(DT)^2 */
/* Increasing COURANT speeds up the simulation, but decreases accuracy */
/* For similar wave forms, COURANT^2*GAMMA should be kept constant */

#define ADD_OSCILLATING_SOURCE 1        /* set to 1 to add an oscillating wave source */
#define OSCILLATING_SOURCE_PERIOD 50     /* period of oscillating source */
#define ALTERNATE_OSCILLATING_SOURCE 1  /* set to 1 to alternate sign of oscillating source */

/* Boundary conditions, see list in global_pdes.c  */

#define B_COND 2

/* Parameters for length and speed of simulation */

#define NSTEPS 2700       /* number of frames of movie */
#define NVID 6            /* number of iterations between images displayed on screen */
#define NSEG 1000         /* number of segments of boundary */
#define INITIAL_TIME 0      /* time after which to start saving frames */
#define BOUNDARY_WIDTH 2    /* width of billiard boundary */
#define PRINT_SPEED 0       /* print speed of moving source */

#define PAUSE 200       /* number of frames after which to pause */
#define PSLEEP 1         /* sleep time during pause */
#define SLEEP1  1        /* initial sleeping time */
#define SLEEP2  1        /* final sleeping time */
#define MID_FRAMES 20    /* number of still frames between parts of two-part movie */
#define END_FRAMES 100    /* number of still frames at end of movie */
#define FADE 1           /* set to 1 to fade at end of movie */

/* Parameters of initial condition */

#define INITIAL_AMP 0.25            /* amplitude of initial condition */
#define INITIAL_VARIANCE 0.00015    /* variance of initial condition */
#define INITIAL_WAVELENGTH  0.0075  /* wavelength of initial condition */

/* Plot type, see list in global_pdes.c  */

#define PLOT 0

#define PLOT_B 5        /* plot type for second movie */

/* Color schemes */

#define COLOR_PALETTE 17    /* Color palette, see list in global_pdes.c  */
#define COLOR_PALETTE_B 13    /* Color palette, see list in global_pdes.c  */

#define BLACK 1          /* background */

#define COLOR_SCHEME 3   /* choice of color scheme, see list in global_pdes.c  */

#define SCALE 0          /* set to 1 to adjust color scheme to variance of field */
#define SLOPE 0.75       /* sensitivity of color on wave amplitude */
#define PHASE_FACTOR 1.0       /* factor in computation of phase in color scheme P_3D_PHASE */
#define PHASE_SHIFT 0.0      /* shift of phase in color scheme P_3D_PHASE */
#define ATTENUATION 0.0  /* exponential attenuation coefficient of contrast with time */
#define E_SCALE 75.0     /* scaling factor for energy representation */
#define LOG_SCALE 1.0     /* scaling factor for energy log representation */
#define LOG_SHIFT 3.5     /* shift of colors on log scale */
#define FLUX_SCALE 5.0e3    /* scaling factor for enegy flux represtnation */
#define RESCALE_COLOR_IN_CENTER 0   /* set to 1 to decrease color intentiy in the center (for wave escaping ring) */

#define COLORHUE 260     /* initial hue of water color for scheme C_LUM */
#define COLORDRIFT 0.0   /* how much the color hue drifts during the whole simulation */
#define LUMMEAN 0.5      /* amplitude of luminosity variation for scheme C_LUM */
#define LUMAMP 0.3       /* amplitude of luminosity variation for scheme C_LUM */
#define HUEMEAN 180.0    /* mean value of hue for color scheme C_HUE */
#define HUEAMP -180.0      /* amplitude of variation of hue for color scheme C_HUE */

#define DRAW_COLOR_SCHEME 1    /* set to 1 to plot the color scheme */
#define COLORBAR_RANGE 2.0     /* scale of color scheme bar */
#define COLORBAR_RANGE_B 0.1   /* scale of color scheme bar for 2nd part */
#define ROTATE_COLOR_SCHEME 0   /* set to 1 to draw color scheme horizontally */

#define SAVE_TIME_SERIES 0      /* set to 1 to save wave time series at a point */

#define NXMAZE 8      /* width of maze */
#define NYMAZE 32      /* height of maze */
#define MAZE_MAX_NGBH 5     /* max number of neighbours of maze cell */
#define RAND_SHIFT 0        /* seed of random number generator */
#define MAZE_XSHIFT 0.0     /* horizontal shift of maze */

/* for compatibility with sub_wave and sub_maze */
#define ADD_POTENTIAL 0
#define POT_MAZE 7
#define POTENTIAL 0
/* end of constants only used by sub_wave and sub_maze */



/* only for compatibility with wave_common.c */
#define TWOSPEEDS 0          /* set to 1 to replace hardcore boundary by medium with different speed */
#define VARIABLE_IOR 0      /* set to 1 for a variable index of refraction */
#define IOR 4               /* choice of index of refraction, see list in global_pdes.c */
#define IOR_TOTAL_TURNS 1.5 /* total angle of rotation for IOR_PERIODIC_WELLS_ROTATING */
#define MANDEL_IOR_SCALE -0.05   /* parameter controlling dependence of IoR on Mandelbrot escape speed */
#define OMEGA 0.005        /* frequency of periodic excitation */
#define COURANT 0.08       /* Courant number */
#define COURANTB 0.03      /* Courant number in medium B */
#define INITIAL_AMP 0.5         /* amplitude of initial condition */
#define INITIAL_VARIANCE 0.0002  /* variance of initial condition */
#define INITIAL_WAVELENGTH  0.1  /* wavelength of initial condition */
#define VSCALE_ENERGY 200.0       /* additional scaling factor for color scheme P_3D_ENERGY */
#define PHASE_FACTOR 20.0       /* factor in computation of phase in color scheme P_3D_PHASE */
#define PHASE_SHIFT 0.0      /* shift of phase in color scheme P_3D_PHASE */
#define OSCILLATION_SCHEDULE 0  /* oscillation schedule, see list in global_pdes.c */
#define AMPLITUDE 0.8      /* amplitude of periodic excitation */ 
#define ACHIRP 0.2        /* acceleration coefficient in chirp */
#define DAMPING 0.0        /* damping of periodic excitation */
#define COMPARISON 0        /* set to 1 to compare two different patterns (beta) */
#define B_DOMAIN_B 20       /* second domain shape, for comparisons */
#define CIRCLE_PATTERN_B 0  /* second pattern of circles or polygons */
#define FLUX_WINDOW 20      /* averaging window for energy flux */
#define ADD_WAVE_PACKET_SOURCES 1       /* set to 1 to add several sources emitting wave packets */
#define WAVE_PACKET_SOURCE_TYPE 1       /* type of wave packet sources */
#define N_WAVE_PACKETS 15               /* number of wave packets */
#define WAVE_PACKET_RADIUS 20            /* radius of wave packets */
/* end of constants added only for compatibility with wave_common.c */


double u_3d[2] = {0.75, -0.45};     /* projections of basis vectors for REP_AXO_3D representation */
double v_3d[2] = {-0.75, -0.45};
double w_3d[2] = {0.0, 0.015};
double light[3] = {0.816496581, 0.40824829, 0.40824829};      /* vector of "light" direction for P_3D_ANGLE color scheme */
double observer[3] = {8.0, 8.0, 10.0};    /* location of observer for REP_PROJ_3D representation */ 
int reset_view = 0;         /* switch to reset 3D view parameters (for option ROTATE_VIEW) */

#define Z_SCALING_FACTOR 1.25  /* overall scaling factor of z axis for REP_PROJ_3D representation */
#define XY_SCALING_FACTOR 1.7  /* overall scaling factor for on-screen (x,y) coordinates after projection */
#define ZMAX_FACTOR 1.0        /* max value of z coordinate for REP_PROJ_3D representation */
#define XSHIFT_3D 0.0          /* overall x shift for REP_PROJ_3D representation */
#define YSHIFT_3D -0.1         /* overall y shift for REP_PROJ_3D representation */
#define BORDER_PADDING 0       /* distance from boundary at which to plot points, to avoid boundary effects due to gradient */

/* For debugging purposes only */
#define FLOOR 1         /* set to 1 to limit wave amplitude to VMAX */
#define VMAX 10.0        /* max value of wave amplitude */
#define TEST_GRADIENT 0 /* print norm squared of gradient */

#define REFRESH_B (ZPLOT_B != ZPLOT)||(CPLOT_B != CPLOT)    /* to save computing time, to be improved */
#define COMPUTE_WRAP_ANGLE ((WRAP_ANGLE)&&((cplot == Z_ANGLE_GRADIENT)||(cplot == Z_ANGLE_GRADIENTX)||(cplot == Z_ARGUMENT)||(cplot == Z_ANGLE_GRADIENTX)))
#define PRINT_PARAMETERS ((PRINT_TIME)||(PRINT_VISCOSITY)||(PRINT_RPSLZB)||(PRINT_PROBABILITIES)||(PRINT_NOISE)||(PRINT_FLOW_SPEED))
#define COMPUTE_PRESSURE ((ZPLOT == Z_EULER_PRESSURE)||(CPLOT == Z_EULER_PRESSURE)||(ZPLOT_B == Z_EULER_PRESSURE)||(CPLOT_B == Z_EULER_PRESSURE))

#define ASYM_SPEED_COLOR (VMEAN_SPEED == 0.0)

#include "global_pdes.c"
#include "global_3d.c"          /* constants and global variables */

#include "sub_maze.c"
#include "sub_wave.c"
#include "wave_common.c"        /* common functions for wave_billiard, wave_comparison, etc */

#include "sub_wave_3d_rde.c"    /* should be later replaced by sub_wave_rde.c */

#include "sub_rde.c"    


double f_aharonov_bohm(double r2)
/* radial part of Aharonov-Bohm vector potential */
{
    double r02 = AB_RADIUS*AB_RADIUS;
    
    if (r2 > r02) return(-0.25*r02/r2);
    else return(0.25*(r2 - 2.0*r02)/r02);

//     if (r2 > r02) return(1.0/r2);
//     else return((2.0*r02 - r2)/(r02*r02));    
}

double potential(int i, int j)
/* compute potential (e.g. for Schrödinger equation), or potential part if there is a magnetic field */
{
    double x, y, xy[2], r, small = 1.0e-1, kx, ky, lx = XMAX - XMIN, r1, r2, r3, f;
    int rect;
    
    ij_to_xy(i, j, xy);
    x = xy[0];
    y = xy[1];
    
    switch (POTENTIAL) {
        case (POT_HARMONIC):
        {
            return (K_HARMONIC*(x*x + y*y));
        }
        case (POT_COULOMB):
        {
//             r = module2(x, y);
            r = sqrt(x*x + y*y + small*small);
//             if (r < small) r = small;
            return (-K_COULOMB/r);
        }
        case (POT_PERIODIC):
        {
            kx = 4.0*DPI/(XMAX - XMIN);
            ky = 2.0*DPI/(YMAX - YMIN);
            return(-K_HARMONIC*cos(kx*x)*cos(ky*y));
        }
        case (POT_FERMIONS):
        {
            r = sqrt((x-y)*(x-y) + small*small);
            return (-K_COULOMB/r);
        }
        case (POT_FERMIONS_PERIODIC):
        {
            r1 = sqrt((x-y)*(x-y) + small*small);
            r2 = sqrt((x-lx-y)*(x-lx-y) + small*small);
            r3 = sqrt((x+lx-y)*(x+lx-y) + small*small);
//             r = r/3.0;
            return (-0.5*K_COULOMB*(1.0/r1 + 1.0/r2 + 1.0/r3));
        }
        case (VPOT_CONSTANT_FIELD):
        {
            return (K_HARMONIC*(x*x + y*y));        /* magnetic field strength b is chosen such that b^2 = K_HARMONIC */
        }
        case (VPOT_AHARONOV_BOHM):
        {
            r2 = x*x + y*y;
            f = f_aharonov_bohm(r2);
            return (B_FIELD*B_FIELD*f*f*r2);    /* magnetic field strength b is chosen such that b^2 = K_HARMONIC */
//             return (K_HARMONIC*f);    /* magnetic field strength b is chosen such that b^2 = K_HARMONIC */
        }
        case (POT_MAZE):
        {
            for (rect=0; rect<npolyrect; rect++)
                if (ij_in_polyrect(i, j, polyrect[rect])) return(V_MAZE);
            return(0.0);    
        }
        default:
        {
            return(0.0);
        }
    }
}   


void compute_vector_potential(int i, int j, double *ax, double *ay)
/* initialize the vector potential, for Schrodinger equation in a magnetic field */
{
    double x, y, xy[2], r2, f;
    
    ij_to_xy(i, j, xy);
    x = xy[0];
    y = xy[1];
    
    switch (POTENTIAL) {
        case (VPOT_CONSTANT_FIELD):
        {
            *ax = B_FIELD*y;
            *ay = -B_FIELD*x;
            break;
        }
        case (VPOT_AHARONOV_BOHM):
        {
            r2 = x*x + y*y;
            f = f_aharonov_bohm(r2);
            *ax = B_FIELD*y*f;
            *ay = -B_FIELD*x*f;
            break;
        }
        default:
        {
            *ax = 0.0;
            *ay = 0.0;
        }
    }
}

void compute_gfield(int i, int j, double *gx, double *gy)
/* initialize the exterior field, for the compressible Euler equation */
{
    double x, y, xy[2], r, f, a = 0.4, x1, y1, hx, hy, h;
    
    ij_to_xy(i, j, xy);
    x = xy[0];
    y = xy[1];
    
    switch (FORCE_FIELD) {
        case (GF_VERTICAL):
        {
            *gx = 0.0;
            *gy = -G_FIELD;
            break;
        }
        case (GF_CIRCLE):
        {
            r = module2(x,y) + 1.0e-2;
            f = 0.5*(1.0 - tanh(BC_STIFFNESS*(r - LAMBDA))); 
            *gx = G_FIELD*f*x/r;
            *gy = G_FIELD*f*y/r;
            break;
        }
        case (GF_ELLIPSE):
        {
            r = module2(x/LAMBDA,y/MU) + 1.0e-2;
            f = 0.5*(1.0 - tanh(BC_STIFFNESS*(r - 1.0))); 
            *gx = G_FIELD*f*x/(LAMBDA*LAMBDA);
            *gy = G_FIELD*f*y/(MU*MU);
            break;
        }
        case (GF_AIRFOIL):
        {
            y1 = y + a*x*x;
            r = module2(x/LAMBDA,y1/MU) + 1.0e-2;
            f = 0.5*(1.0 - tanh(BC_STIFFNESS*(r - 1.0))); 
            *gx = G_FIELD*f*(x/(LAMBDA*LAMBDA) + a*y1/(MU*MU));
            *gy = G_FIELD*f*y1/(MU*MU);
            break;
        }
        case (GF_WING):
        {
            if (x >= LAMBDA)
            {
                *gx = 0.0;
                *gy = 0.0;
            }
            else
            {
                x1 = 1.0 - x/LAMBDA;
                if (x1 < 0.1) x1 = 0.1;
                y1 = y + a*x*x;
                r = module2(x/LAMBDA,y1/(MU*x1)) + 1.0e-2;
                f = 0.5*(1.0 - tanh(BC_STIFFNESS*(r - 1.0))); 
                *gx = G_FIELD*f*(x/(LAMBDA*LAMBDA) + 2.0*a*x*y1/(MU*MU*x1*x1) - y1*y1/(MU*MU*x1*x1*x1));
                *gy = G_FIELD*f*y1/(MU*MU*x1*x1);
//                 *gx = 0.1*G_FIELD*f*(x/(LAMBDA*LAMBDA) + 2.0*a*x*y1/(MU*MU*x1*x1) - y1*y1/(MU*MU*x1*x1*x1));
//                 *gy = 0.1*G_FIELD*f*y1/(MU*MU*x1*x1);
//                 hx = x/(LAMBDA*LAMBDA) + 2.0*a*x*y1/(MU*MU*x1*x1) - y1*y1/(MU*MU*x1*x1*x1);
//                 hy = y1/(MU*MU*x1*x1);
//                 h = module2(hx, hy) + 1.0e-2;
//                 *gx = G_FIELD*f*hx/h;
//                 *gy = G_FIELD*f*hy/h;
            }
            break;
        }
        default:
        {
            *gx = 0.0;
            *gy = 0.0;
        }
    }
}
void initialize_potential(double potential_field[NX*NY])
/* initialize the potential field, e.g. for the Schrödinger equation */
{
    int i, j;
    
    #pragma omp parallel for private(i,j)
    for (i=0; i<NX; i++){
        for (j=0; j<NY; j++){
            potential_field[i*NY+j] = potential(i,j);
        }
    }
}

void initialize_vector_potential(double vpotential_field[2*NX*NY])
/* initialize the potential field, e.g. for the Schrödinger equation */
{
    int i, j;
    
    #pragma omp parallel for private(i,j)
    for (i=0; i<NX; i++){
        for (j=0; j<NY; j++){
            compute_vector_potential(i, j, &vpotential_field[i*NY+j], &vpotential_field[NX*NY+i*NY+j]);
        }
    }
}

void initialize_gfield(double gfield[2*NX*NY], double bc_field[NX*NY])
/* initialize the exterior field, e.g. for the compressible Euler equation */
{
    int i, j;
    double dx, dy;
    
    if (FORCE_FIELD == GF_COMPUTE_FROM_BC)
    {
        dx = (XMAX - XMIN)/(double)NX;
        dy = (YMAX - YMIN)/(double)NY;

        #pragma omp parallel for private(i,j)
        for (i=1; i<NX-1; i++){
            for (j=1; j<NY-1; j++){
                gfield[i*NY+j] = G_FIELD*(bc_field[(i+1)*NY+j] - bc_field[(i-1)*NY+j])/dx;
                gfield[NX*NY+i*NY+j] = G_FIELD*(bc_field[i*NY+j+1] - bc_field[i*NY+j-1])/dy;
                printf("gfield at (%i,%i): (%.3lg, %.3lg)\n", i, j, gfield[i*NY+j], gfield[NX*NY+i*NY+j]);
            }
        }
        
        /* boundaries */
        for (i=0; i<NX; i++)
        {
            gfield[i*NY] = 0.0;
            gfield[NX*NY+i*NY] = 0.0;
            gfield[i*NY+NY-1] = 0.0;
            gfield[NX*NY+i*NY+NY-1] = 0.0;
        }
        for (j=0; j<NY; j++)
        {
            gfield[j] = 0.0;
            gfield[NX*NY+j] = 0.0;
            gfield[(NX-1)*NY+j] = 0.0;
            gfield[NX*NY+(NX-1)*NY+j] = 0.0;            
        }
    }
    
    else
    {
        #pragma omp parallel for private(i,j)
        for (i=0; i<NX; i++){
            for (j=0; j<NY; j++){
                compute_gfield(i, j, &gfield[i*NY+j], &gfield[NX*NY+i*NY+j]);
            }
        }
    }
}

void evolve_wave_half(double *phi_in[NFIELDS], double *phi_out[NFIELDS], short int xy_in[NX*NY], 
                      double potential_field[NX*NY], double vector_potential_field[2*NX*NY], 
                      double gfield[2*NX*NY], t_rde rde[NX*NY])
/* time step of field evolution */
{
    int i, j, k, iplus, iminus, jplus, jminus, ropening;
    double x, y, z, deltax, deltay, deltaz, rho, rhox, rhoy, pot, u, v, ux, uy, vx, vy, test = 0.0, dx, dy, xy[2], padding, a;
    double *delta_phi[NLAPLACIANS], *nabla_phi, *nabla_psi, *nabla_omega, *delta_vorticity, *delta_pressure, *delta_p, *delta_u, *delta_v, *nabla_rho, *nabla_u, *nabla_v;
//     double u_bc[NY], v_bc[NY]; 
    static double invsqr3 = 0.577350269;    /* 1/sqrt(3) */
    static double stiffness = 2.0;     /* stiffness of Poisson equation solver */
    static int smooth = 0, y_channels, imin, imax, first = 1;
    
    if (first)  /* for D_MAZE_CHANNELS boundary conditions in Euler equation */
    {
        ropening = (NYMAZE+1)/2;
        padding = 0.02;
        dy = (YMAX - YMIN - 2.0*padding)/(double)(NYMAZE);
        y = YMIN + 0.02 + dy*((double)ropening);
        x = YMAX - padding + MAZE_XSHIFT;
        xy_to_pos(x, y, xy);
        y_channels = xy[1] - 5;
        if ((B_DOMAIN == D_MAZE_CHANNELS)||(OBSTACLE_GEOMETRY == D_MAZE_CHANNELS))
        {
            imax = xy[0] + 2;
            x = YMIN + padding + MAZE_XSHIFT;
            xy_to_pos(x, y, xy);
            imin = xy[0] - 2;
            if (imin < 5) imin = 5;
        }
        else if (OBSTACLE_GEOMETRY == D_TESLA)
        {
            imin = 0;
            imax = NX;
            y = -a;
            xy_to_pos(XMIN, y, xy);
            y_channels = xy[1]; 
            printf("y_channels = %i\n", y_channels);
        }
        else
        {
            imin = 0;
            imax = NX;
        }
        first = 0;
    }
    
    for (i=0; i<NLAPLACIANS; i++) delta_phi[i] = (double *)malloc(NX*NY*sizeof(double));
    
    if (COMPUTE_PRESSURE) 
    {
        delta_pressure = (double *)malloc(NX*NY*sizeof(double));
        delta_p = (double *)malloc(NX*NY*sizeof(double));
    }
    
    /* compute the Laplacian of phi */
    for (i=0; i<NLAPLACIANS; i++) compute_laplacian_rde(phi_in[i], delta_phi[i], xy_in);
    
    if (COMPUTE_PRESSURE) compute_laplacian_rde(phi_in[2], delta_pressure, xy_in);
    
    /* compute the gradient of phi if there is a magnetic field */
    if (ADD_MAGNETIC_FIELD) 
    {
        nabla_phi = (double *)malloc(2*NX*NY*sizeof(double));
        nabla_psi = (double *)malloc(2*NX*NY*sizeof(double));
        compute_gradient_xy(phi_in[0], nabla_phi);
        compute_gradient_xy(phi_in[1], nabla_psi);
    }
    
    /* compute gradients of stream function and vorticity for Euler equation */
    if (RDE_EQUATION == E_EULER_INCOMP)
    {
        nabla_psi = (double *)malloc(2*NX*NY*sizeof(double));
        nabla_omega = (double *)malloc(2*NX*NY*sizeof(double));
        compute_gradient_euler(phi_in[0], nabla_psi, EULER_GRADIENT_YSHIFT);
        compute_gradient_euler(phi_in[1], nabla_omega, 0.0);
        
        if (COMPUTE_PRESSURE) compute_pressure_laplacian(phi_in, delta_p);
        
        dx = (XMAX-XMIN)/((double)NX);
        dy = (YMAX-YMIN)/((double)NY);
        
        if (SMOOTHEN_VORTICITY)     /* beta: try to reduce formation of ripples */
        {
            if (smooth == 0)
            {
                delta_vorticity = (double *)malloc(NX*NY*sizeof(double));
                compute_laplacian_rde(phi_in[1], delta_vorticity, xy_in); 
//                 #pragma omp parallel for private(i,delta_vorticity)
                for (i=0; i<NX*NY; i++) phi_in[1][i] += intstep*SMOOTH_FACTOR*delta_vorticity[i];
                free(delta_vorticity);
            }
            smooth++;
            if (smooth >= SMOOTHEN_PERIOD) smooth = 0;
        }
    }
    
    /* compute gradients of fields for compressible Euler equation */
    else if (RDE_EQUATION == E_EULER_COMP)
    {
        nabla_rho = (double *)malloc(2*NX*NY*sizeof(double));
//         nabla_u = (double *)malloc(2*NX*NY*sizeof(double));
//         nabla_v = (double *)malloc(2*NX*NY*sizeof(double));
        compute_gradient_euler_test(phi_in[0], nabla_rho, xy_in);
        compute_velocity_gradients(phi_in, rde);
//         compute_gradient_euler_test(phi_in[1], nabla_u, xy_in);
//         compute_gradient_euler_test(phi_in[2], nabla_v, xy_in);
        
        if (SMOOTHEN_VELOCITY)     /* beta: try to reduce formation of ripples */
        {
            if (smooth == 0)
            {
                delta_u = (double *)malloc(NX*NY*sizeof(double));
                delta_v = (double *)malloc(NX*NY*sizeof(double));
                compute_laplacian_rde(phi_in[1], delta_u, xy_in); 
                compute_laplacian_rde(phi_in[2], delta_v, xy_in); 
                #pragma omp parallel for private(i)
                for (i=0; i<NX*NY; i++) phi_in[1][i] += intstep*SMOOTH_FACTOR*delta_u[i];
                #pragma omp parallel for private(i)
                for (i=0; i<NX*NY; i++) phi_in[2][i] += intstep*SMOOTH_FACTOR*delta_v[i];
                free(delta_u);
                free(delta_v);
            }
            smooth++;
            if (smooth >= SMOOTHEN_PERIOD) smooth = 0;
        }
    } 
    
    if (TEST_GRADIENT) {
        test = 0.0;
        for (i=0; i<2*NX*NY; i++){
            test += nabla_v[i]*nabla_v[i];
//             test += nabla_omega[i]*nabla_omega[i];
//             test += nabla_psi[i]*nabla_psi[i];
        }
        printf("nabla square = %.5lg\n", test/((double)NX*NY));
    }
    
    
    #pragma omp parallel for private(i,j,k,x,y,z,deltax,deltay,deltaz,rho)
    for (i=imin; i<imax; i++){
        for (j=0; j<NY; j++){
            if (xy_in[i*NY+j]) switch (RDE_EQUATION){
                case (E_HEAT):
                {
                    deltax = viscosity*delta_phi[0][i*NY+j];
                    phi_out[0][i*NY+j] = phi_in[0][i*NY+j] + intstep*deltax;
                    break;
                }
                case (E_ALLEN_CAHN):
                {
                    x = phi_in[0][i*NY+j];
                    deltax = viscosity*delta_phi[0][i*NY+j];
                    phi_out[0][i*NY+j] = phi_in[0][i*NY+j] + intstep*(deltax + x*(1.0-x*x));
                    break;
                }
                case (E_CAHN_HILLIARD):
                {
                    /* TO DO */
                    break;
                }
                case (E_FHN):
                {
                    x = phi_in[0][i*NY+j];
                    y = phi_in[1][i*NY+j];
                    deltax = viscosity*delta_phi[0][i*NY+j];
                    phi_out[0][i*NY+j] = phi_in[0][i*NY+j] + intstep*(deltax + x*(1.0-x*x) + y);
                    phi_out[1][i*NY+j] = phi_in[0][i*NY+j] + intstep*EPSILON*(- invsqr3 - FHNC - FHNA*x);
                    break;
                }
                case (E_RPS):
                {
                    x = phi_in[0][i*NY+j];
                    y = phi_in[1][i*NY+j];
                    z = phi_in[2][i*NY+j];
                    rho = x + y + z;
                    deltax = viscosity*delta_phi[0][i*NY+j];
                    deltay = viscosity*delta_phi[1][i*NY+j];
                    deltaz = viscosity*delta_phi[2][i*NY+j];
                
                    phi_out[0][i*NY+j] = x + intstep*(deltax + x*(1.0 - rho - RPSA*y));
                    phi_out[1][i*NY+j] = y + intstep*(deltay + y*(1.0 - rho - RPSA*z));
                    phi_out[2][i*NY+j] = z + intstep*(deltaz + z*(1.0 - rho - RPSA*x));
                    break;
                }
                case (E_RPSLZ):
                {
                    rho = 0.0;
                    for (k=0; k<5; k++) rho += phi_in[k][i*NY+j];
                    
                    for (k=0; k<5; k++) 
                    {
                        x = phi_in[k][i*NY+j];
                        y = phi_in[(k+1)%5][i*NY+j];
                        z = phi_in[(k+3)%5][i*NY+j];
                        phi_out[k][i*NY+j] = x + intstep*(delta_phi[k][i*NY+j] + x*(1.0 - rho - RPSA*y - rpslzb*z));
                    }
                    break;
                }
                case (E_SCHRODINGER):
                {
                    phi_out[0][i*NY+j] = phi_in[0][i*NY+j] - intstep*delta_phi[1][i*NY+j];
                    phi_out[1][i*NY+j] = phi_in[1][i*NY+j] + intstep*delta_phi[0][i*NY+j];
                    if ((ADD_POTENTIAL)||(ADD_MAGNETIC_FIELD))
                    {
                        pot = potential_field[i*NY+j];
                        phi_out[0][i*NY+j] += intstep*pot*phi_in[1][i*NY+j];
                        phi_out[1][i*NY+j] -= intstep*pot*phi_in[0][i*NY+j];
                    }
                    if (ADD_MAGNETIC_FIELD)
                    {
                        vx = vector_potential_field[i*NY+j];
                        vy = vector_potential_field[NX*NY+i*NY+j];
                        phi_out[0][i*NY+j] -= 2.0*intstep*(vx*nabla_phi[i*NY+j] + vy*nabla_phi[NX*NY+i*NY+j]);
                        phi_out[1][i*NY+j] -= 2.0*intstep*(vx*nabla_psi[i*NY+j] + vy*nabla_psi[NX*NY+i*NY+j]);
                    }
                    break;
                }
                case (E_EULER_INCOMP):
                {
                    phi_out[0][i*NY+j] = phi_in[0][i*NY+j] + intstep*stiffness*(delta_phi[0][i*NY+j] + phi_in[1][i*NY+j]*dx*dx);
//                     phi_out[0][i*NY+j] += intstep*EULER_GRADIENT_YSHIFT;
                    phi_out[1][i*NY+j] = phi_in[1][i*NY+j] - intstep*K_EULER*(nabla_omega[i*NY+j]*nabla_psi[NX*NY+i*NY+j]);
                    phi_out[1][i*NY+j] += intstep*K_EULER*(nabla_omega[NX*NY+i*NY+j]*nabla_psi[i*NY+j]);
                        
                    if (COMPUTE_PRESSURE)
                    {
                        phi_out[2][i*NY+j] = phi_in[2][i*NY+j] + intstep*stiffness*(delta_pressure[i*NY+j] - delta_p[i*NY+j]);
                        phi_out[2][i*NY+j] *= exp(-2.0e-3);
                    }
                    break;
                }
                case (E_EULER_COMP):
                {
                    rho = phi_in[0][i*NY+j];
                    if (rho == 0.0) rho = 1.0e-1;
                    u = phi_in[1][i*NY+j];
                    v = phi_in[2][i*NY+j];
                    rhox = nabla_rho[i*NY+j];
                    rhoy = nabla_rho[NX*NY+i*NY+j];
                    
                    ux = rde[i*NY+j].dxu;
                    uy = rde[i*NY+j].dyu;
                    vx = rde[i*NY+j].dxv;
                    vy = rde[i*NY+j].dyv;
                    
                    phi_out[0][i*NY+j] = rho - intstep*(u*rhox + v*rhoy + rho*(ux + vy));
                    phi_out[1][i*NY+j] = u - intstep*(u*ux + v*uy + K_EULER_INC*rhox/rho);
                    phi_out[2][i*NY+j] = v - intstep*(u*vx + v*vy + K_EULER_INC*rhoy/rho);
                    
                    if (ADD_FORCE_FIELD)
                    {
                        phi_out[1][i*NY+j] += intstep*gfield[i*NY+j];
                        phi_out[2][i*NY+j] += intstep*gfield[NX*NY+i*NY+j];
                    }
                    break;
                }
            }
        }
    }
    
    /* in-flow/out-flow b.c. for incompressible Euler equation */
    if (((RDE_EQUATION == E_EULER_INCOMP)||(RDE_EQUATION == E_EULER_COMP))&&(IN_OUT_FLOW_BC > 0))
    {
        switch (IN_OUT_FLOW_BC) {
            case (BCE_LEFT):
            {
                set_boundary_laminar_flow(flow_speed, LAMINAR_FLOW_MODULATION, 0.02, 0.1, 1.0, 0.0, 0.1, phi_out, xy_in, 0, 5, 0, NY, IN_OUT_BC_FACTOR); 
                break;
            }
            case (BCE_TOPBOTTOM):
            {
                set_boundary_laminar_flow(flow_speed, LAMINAR_FLOW_MODULATION, 0.02, LAMINAR_FLOW_YPERIOD, 1.0, -0.1, 0.1, phi_out, xy_in, 0, NX, 0, 10, IN_OUT_BC_FACTOR);
                set_boundary_laminar_flow(flow_speed, LAMINAR_FLOW_MODULATION, 0.02, LAMINAR_FLOW_YPERIOD, 1.0, -0.1, 0.1, phi_out, xy_in, 0, NX, NY-10, NY, IN_OUT_BC_FACTOR);
                break;
            }
            case (BCE_TOPBOTTOMLEFT):
            {
                set_boundary_laminar_flow(flow_speed, LAMINAR_FLOW_MODULATION, 0.02, LAMINAR_FLOW_YPERIOD, 1.0, -0.1, 0.1, phi_out, xy_in, 0, NX, 0, 10, IN_OUT_BC_FACTOR);
                set_boundary_laminar_flow(flow_speed, LAMINAR_FLOW_MODULATION, 0.02, LAMINAR_FLOW_YPERIOD, 1.0, -0.1, 0.1, phi_out, xy_in, 0, NX, NY-10, NY, IN_OUT_BC_FACTOR);
                set_boundary_laminar_flow(flow_speed, LAMINAR_FLOW_MODULATION, 0.02, LAMINAR_FLOW_YPERIOD, 1.0, -0.1, 0.1, phi_out, xy_in, 0, 2, 0, NY, IN_OUT_BC_FACTOR); 
                break;
            }
            case (BCE_CHANNELS):
            {
                set_boundary_laminar_flow(flow_speed, LAMINAR_FLOW_MODULATION, 0.02, LAMINAR_FLOW_YPERIOD, 1.0, 0.0, 0.1, phi_out, xy_in, 0, imin+5, NY - y_channels, y_channels, IN_OUT_BC_FACTOR);  
                set_boundary_laminar_flow(flow_speed, LAMINAR_FLOW_MODULATION, 0.02, LAMINAR_FLOW_YPERIOD, 1.0, 0.0, 0.1, phi_out, xy_in, imax-5, NX - 1, NY- y_channels, y_channels, IN_OUT_BC_FACTOR); 
//                 set_boundary_laminar_flow(flow_speed, LAMINAR_FLOW_MODULATION, 0.02, LAMINAR_FLOW_YPERIOD, 1.0, 0.0, 0.1, phi_out, xy_in, imin-5, imin+10, NY - y_channels, y_channels);  
//                 set_boundary_laminar_flow(flow_speed, LAMINAR_FLOW_MODULATION, 0.02, LAMINAR_FLOW_YPERIOD, 1.0, 0.0, 0.1, phi_out, xy_in, imax-10, imax+5, NY- y_channels, y_channels); 
                break;
            }
            case (BCE_MIDDLE_STRIP):
            {
                set_boundary_laminar_flow(flow_speed, LAMINAR_FLOW_MODULATION, 0.02, LAMINAR_FLOW_YPERIOD, 1.0, 0.0, 0.1, phi_out, xy_in, 0, NX, NY/2 - 10, NY/2 + 10, IN_OUT_BC_FACTOR);
                set_boundary_laminar_flow(flow_speed, LAMINAR_FLOW_MODULATION, 0.02, LAMINAR_FLOW_YPERIOD, 1.0, 0.0, 0.1, phi_out, xy_in, 0, 2, 0, NY, IN_OUT_BC_FACTOR); 
                set_boundary_laminar_flow(flow_speed, LAMINAR_FLOW_MODULATION, 0.02, LAMINAR_FLOW_YPERIOD, 1.0, 0.0, 0.1, phi_out, xy_in, NX-2, NX, 0, NY, IN_OUT_BC_FACTOR); 
                break;
            }
        }
    }
        
    if (TEST_GRADIENT) {
//         test = 0.0;
//         for (i=0; i<NX*NY; i++){
//             test += delta_phi[0][i] + phi_out[1][i]*dx*dx;
//         }
//         printf("Delta psi + omega = %.5lg\n", test/((double)NX*NY));
    }
                
    if (FLOOR) for (i=0; i<NX; i++){
        for (j=0; j<NY; j++){
            if (xy_in[i*NY+j] != 0) for (k=0; k<NFIELDS; k++)
            {
                if (phi_out[k][i*NY+j] > VMAX) phi_out[k][i*NY+j] = VMAX;
                if (phi_out[k][i*NY+j] < -VMAX) phi_out[k][i*NY+j] = -VMAX;
            }
        }
    }
    
    for (i=0; i<NLAPLACIANS; i++) free(delta_phi[i]);
    
    if (ADD_MAGNETIC_FIELD) 
    {
        free(nabla_phi);
        free(nabla_psi);
    }
    
    if (RDE_EQUATION == E_EULER_INCOMP) 
    {
        free(nabla_psi);
        free(nabla_omega);
    }
    else if (RDE_EQUATION == E_EULER_COMP)
    {
        free(nabla_rho);
//         free(nabla_u);
//         free(nabla_v);
    }
    
    if (COMPUTE_PRESSURE) 
    {
        free(delta_pressure);
        free(delta_p);
    }
}

void evolve_wave(double *phi[NFIELDS], double *phi_tmp[NFIELDS], short int xy_in[NX*NY], 
                 double potential_field[NX*NY], double vector_potential_field[2*NX*NY], 
                 double gfield[2*NX*NY], t_rde rde[NX*NY])
/* time step of field evolution */
{
    evolve_wave_half(phi, phi_tmp, xy_in, potential_field, vector_potential_field, gfield, rde);
    evolve_wave_half(phi_tmp, phi, xy_in, potential_field, vector_potential_field, gfield, rde);
}


void evolve_tracers(double *phi[NFIELDS], double tracers[2*N_TRACERS*NSTEPS], int time, int nsteps, double step)
/* time steps of tracer particle evolution (for Euler equation) */
{
    int tracer, i, j, t, ij[2], iplus, jplus;
    double x, y, xy[2], vx, vy; 
    
    step = TRACERS_STEP;
    
    for (tracer = 0; tracer < N_TRACERS; tracer++)
    {
        x = tracers[time*2*N_TRACERS + 2*tracer];
        y = tracers[time*2*N_TRACERS + 2*tracer + 1];
        
//         printf("Tracer %i position (%.2f, %.2f)\n", tracer, x, y);
        
        for (t=0; t<nsteps; t++) 
        {
            xy_to_ij_safe(x, y, ij);
            i = ij[0];
            j = ij[1];
            
            switch (RDE_EQUATION) {
                case (E_EULER_INCOMP): 
                {
                    iplus = i + 1;  if (iplus == NX) iplus = 0;
                    jplus = j + 1;  if (jplus == NY) jplus = 0;
        
                    vx = phi[0][i*NY+jplus] - phi[0][i*NY+j];
                    vy = -(phi[0][iplus*NY+j] - phi[0][i*NY+j]);
            
                    if (j == 0) vx += EULER_GRADIENT_YSHIFT;
                    else if (j == NY-1) vx -= EULER_GRADIENT_YSHIFT;
                    break;
                }
                case (E_EULER_COMP):
                {
                    vx = phi[1][i*NY+j];
                    vy = phi[2][i*NY+j];
                    break;
                }
            }
            
//             v = module2(vx, vy);
//             if ((v > 0.0)&&(v < 0.1)) 
//             {
//                 vx = vx*0.1/v;
//                 vy = vy*0.1/v;
//             }
            
//             printf("(i, j) = (%i, %i), Tracer %i velocity (%.6f, %.6f)\n", i, j, tracer, vx, vy);
            
            x += vx*step;
            y += vy*step;
        }
//         printf("Tracer %i velocity (%.2f, %.2f)\n", tracer, vx, vy);
        
        if (x > XMAX) x += (XMIN - XMAX);
        if (x < XMIN) x += (XMAX - XMIN);
        if (y > YMAX) y += (YMIN - YMAX);
        if (y < YMIN) y += (YMAX - YMIN);
        
        if (time+1 < NSTEPS)
        {
            tracers[(time+1)*2*N_TRACERS + 2*tracer] = x;
            tracers[(time+1)*2*N_TRACERS + 2*tracer + 1] = y;
        }
    }
}


void print_level(int level)
{
    double pos[2];
    char message[50];
    
    glColor3f(1.0, 1.0, 1.0);
    sprintf(message, "Level %i", level);
    xy_to_pos(XMIN + 0.1, YMAX - 0.2, pos);
    write_text(pos[0], pos[1], message);
}



void print_parameters(t_rde rde[NX*NY], short int xy_in[NX*NY], double time, short int left, double viscosity, double noise)
{
    char message[100];
    double density, hue, rgb[3], logratio, x, y, pos[2], probas[2];
    static double xbox, xtext, boxwidth, boxheight;
    static int first = 1;
    
    if (first)
    {
        if (WINWIDTH > 1280)
        {
            boxheight = 0.035;
            boxwidth = 0.21;
            if (left)
            {
                xbox = XMIN + 0.4;
                xtext = XMIN + 0.2;
            }
            else
            {
                xbox = XMAX - 0.49;
                xtext = XMAX - 0.65;
//                 xbox = XMAX - 0.39;
//                 xtext = XMAX - 0.55;
            }
        }
        else
        {
            boxwidth = 0.3;
            boxheight = 0.05;
            if (left)
            {
                xbox = XMIN + 0.4;
                xtext = XMIN + 0.1;
            }
            else
            {
                xbox = XMAX - 0.49;
                xtext = XMAX - 0.71;
//                 xbox = XMAX - 0.39;
//                 xtext = XMAX - 0.61;
            }
        }
         
        first = 0;
    }
    
    if (PRINT_PROBABILITIES)
    {
        compute_probabilities(rde, xy_in, probas);
        printf("pleft = %.3lg, pright = %.3lg\n", probas[0], probas[1]);
        
        x = XMIN + 0.15*(XMAX - XMIN);
        y = YMIN + 0.3*(YMAX - YMIN);
        erase_area_hsl(x, y, boxwidth, boxheight, 0.0, 0.9, 0.0);
        glColor3f(1.0, 1.0, 1.0);
        sprintf(message, "Proba %.3f", probas[0]);
        write_text(x, y, message);
        
        x = XMIN + 0.72*(XMAX - XMIN);
        y = YMIN + 0.68*(YMAX - YMIN);
        erase_area_hsl(x, y, boxwidth, boxheight, 0.0, 0.9, 0.0);
        glColor3f(1.0, 1.0, 1.0);
        sprintf(message, "Proba %.3f", probas[1]);
        write_text(x, y, message);
    }
    else
    {
        y = YMAX - 0.1;
        erase_area_hsl(xbox, y + 0.02, boxwidth, boxheight, 0.0, 0.9, 0.0);
        glColor3f(1.0, 1.0, 1.0);
        if (PRINT_TIME) sprintf(message, "Time %.3f", time);
        else if (PRINT_VISCOSITY) sprintf(message, "Viscosity %.3f", viscosity);
        else if (PRINT_RPSLZB) sprintf(message, "b = %.3f", rpslzb);
        else if (PRINT_NOISE) sprintf(message, "noise %.3f", noise);
        else if (PRINT_FLOW_SPEED) sprintf(message, "Speed %.3f", flow_speed);
        if (PLOT_3D) write_text(xtext, y, message);
        else
        {
            xy_to_pos(xtext, y, pos);
            write_text(pos[0], pos[1], message);
        }
    }
}

void draw_color_bar_palette(int plot, double range, int palette, int fade, double fade_value)
{
    double width = 0.14;
//     double width = 0.2;
    
    if (PLOT_3D)
    {
        if (ROTATE_COLOR_SCHEME) 
            draw_color_scheme_palette_3d(XMIN + 0.3, YMIN + 0.1, XMAX - 0.3, YMIN + 0.1 + width, plot, -range, range, palette, fade, fade_value);
//             draw_color_scheme_palette_3d(-1.0, -0.8, XMAX - 0.1, -1.0, plot, -range, range, palette, fade, fade_value);
        else 
            draw_color_scheme_palette_3d(XMAX - 1.5*width, YMIN + 0.1, XMAX - 0.5*width, YMAX - 0.1, plot, -range, range, palette, fade, fade_value);
    }
    else
    {
        if (ROTATE_COLOR_SCHEME) 
            draw_color_scheme_palette_fade(XMIN + 0.8, YMIN + 0.1, XMAX - 0.8, YMIN + 0.1 + width, plot, -range, range, palette, fade, fade_value);
//             draw_color_scheme_palette_fade(-1.0, -0.8, XMAX - 0.1, -1.0, plot, -range, range, palette, fade, fade_value);
        else 
            draw_color_scheme_palette_fade(XMAX - 1.5*width, YMIN + 0.1, XMAX - 0.5*width, YMAX - 0.1, plot, -range, range, palette, fade, fade_value);
    }
}

double noise_schedule(int i)
{
    double ratio;
    
    if (i < NOISE_INITIAL_TIME) return (NOISE_INTENSITY);
    else 
    {
        ratio = (double)(i - NOISE_INITIAL_TIME)/(double)(NSTEPS - NOISE_INITIAL_TIME);
        return (NOISE_INTENSITY*(1.0 + ratio*(NOISE_FACTOR - 1.0)));
    }
}


double viscosity_schedule(int i)
{
    double ratio;
    
    if (i < VISCOSITY_INITIAL_TIME) return (VISCOSITY);
    else 
    {
        ratio = (double)(i - VISCOSITY_INITIAL_TIME)/(double)(NSTEPS - VISCOSITY_INITIAL_TIME);
        return (VISCOSITY*(1.0 + ratio*(VISCOSITY_FACTOR - 1.0)));
    }
}

double rpslzb_schedule(int i)
{
    double ratio;
    
    if (i < RPSLZB_INITIAL_TIME) return (RPSLZB);
    else if (i > NSTEPS - RPSLZB_FINAL_TIME) return(RPSLZB - RPSLZB_CHANGE);
    else 
    {
        ratio = (double)(i - RPSLZB_INITIAL_TIME)/(double)(NSTEPS - RPSLZB_INITIAL_TIME - RPSLZB_FINAL_TIME);
        return (RPSLZB - ratio*RPSLZB_CHANGE);
    }
}

double flow_speed_schedule(int i)
{
    double ratio;
    
    ratio = (double)i/(double)NSTEPS;
    return (IN_OUT_FLOW_MIN_AMP + (IN_OUT_FLOW_AMP - IN_OUT_FLOW_MIN_AMP)*ratio);
}


void viewpoint_schedule(int i)
/* change position of observer */
{
    int j;
    double angle, ca, sa;
    static double observer_initial[3];
    static int first = 1;
    
    if (first)
    {
        for (j=0; j<3; j++) observer_initial[j] = observer[j];
        first = 0;
    }
    
    angle = (ROTATE_ANGLE*DPI/360.0)*(double)i/(double)NSTEPS;
    ca = cos(angle);
    sa = sin(angle);
    observer[0] = ca*observer_initial[0] - sa*observer_initial[1];
    observer[1] = sa*observer_initial[0] + ca*observer_initial[1];
    printf("Angle %.3lg, Observer position (%.3lg, %.3lg, %.3lg)\n", angle, observer[0], observer[1], observer[2]);
}


void animation()
{
    double time = 0.0, scale, dx, var, jangle, cosj, sinj, sqrintstep, 
        intstep0, viscosity_printed, fade_value, noise = NOISE_INTENSITY;
    double *phi[NFIELDS], *phi_tmp[NFIELDS], *potential_field, *vector_potential_field, *tracers, *gfield, *bc_field;
    short int *xy_in;
    int i, j, k, s, nvid, field;
    static int counter = 0;
    t_rde *rde;

    /* Since NX and NY are big, it seemed wiser to use some memory allocation here */
    for (i=0; i<NFIELDS; i++)
    {
        phi[i] = (double *)malloc(NX*NY*sizeof(double));
        phi_tmp[i] = (double *)malloc(NX*NY*sizeof(double));
    }

    xy_in = (short int *)malloc(NX*NY*sizeof(short int));
    rde = (t_rde *)malloc(NX*NY*sizeof(t_rde));
    
    npolyline = init_polyline(MDEPTH, polyline);
    for (i=0; i<npolyline; i++) printf("vertex %i: (%.3f, %.3f)\n", i, polyline[i].x, polyline[i].y);
    
    npolyrect = init_polyrect(polyrect);
    for (i=0; i<npolyrect; i++) printf("polyrect vertex %i: (%.3f, %.3f) - (%.3f, %.3f)\n", i, polyrect[i].x1, polyrect[i].y1, polyrect[i].x2, polyrect[i].y2);

    if (ADD_POTENTIAL) 
    {
        potential_field = (double *)malloc(NX*NY*sizeof(double));
        initialize_potential(potential_field);
    }
    else if (ADD_MAGNETIC_FIELD)
    {
        potential_field = (double *)malloc(NX*NY*sizeof(double));
        vector_potential_field = (double *)malloc(2*NX*NY*sizeof(double));
        initialize_potential(potential_field);
        initialize_vector_potential(vector_potential_field);
    }
    
    if (ADAPT_STATE_TO_BC)
    {
        bc_field = (double *)malloc(NX*NY*sizeof(double));
        initialize_bcfield(bc_field, polyrect);
    }
    if (ADD_FORCE_FIELD)
    {
        gfield = (double *)malloc(2*NX*NY*sizeof(double));
        initialize_gfield(gfield, bc_field);
    }
        
    
//     if (ADD_TRACERS) tracers = (double *)malloc(2*NSTEPS*N_TRACERS*sizeof(double));
    if (ADD_TRACERS) tracers = (double *)malloc(4*NSTEPS*N_TRACERS*sizeof(double));

    dx = (XMAX-XMIN)/((double)NX);
    intstep = DT/(dx*dx);
    
    intstep0 = intstep;
    intstep1 = DT/dx;
    
    viscosity = VISCOSITY;
    
    sqrintstep = sqrt(intstep*(double)NVID);
        
    printf("Integration step %.3lg\n", intstep);

    /* initialize field */
//     init_random(0.5, 0.4, phi, xy_in);
//     init_random(0.0, 0.4, phi, xy_in);
//     init_gaussian(x, y, mean, amplitude, scalex, phi, xy_in)
//     init_coherent_state(-1.2, 0.35, 5.0, -2.0, 0.1, phi, xy_in);
//     add_coherent_state(-0.75, -0.75, 0.0, 5.0, 0.1, phi, xy_in);
//     init_fermion_state(-0.5, 0.5, 2.0, 0.0, 0.1, phi, xy_in);
//     init_boson_state(-0.5, 0.5, 2.0, 0.0, 0.1, phi, xy_in);

//     init_vortex_state(0.1, 0.4, 0.0, 0.3, -0.1, phi, xy_in);
//     add_vortex_state(0.1, -0.4, 0.0, 0.3, 0.1, phi, xy_in);
    
//     init_shear_flow(1.0, 0.02, 0.15, 1, 1, phi, xy_in);
//     init_laminar_flow(flow_speed_schedule(0), LAMINAR_FLOW_MODULATION, LAMINAR_FLOW_YPERIOD, 0.0, phi, xy_in);
//     init_laminar_flow(IN_OUT_FLOW_AMP, LAMINAR_FLOW_MODULATION, 0.02, 0.1, 1.0, 0.0, 0.1, phi, xy_in);
    init_laminar_flow(flow_speed_schedule(0), LAMINAR_FLOW_MODULATION, 0.02, 0.1, 1.0, 0.0, 0.1, phi, xy_in);
    
//     init_shear_flow(-1.0, 0.1, 0.2, 1, 1, 0.2, phi, xy_in);
//     init_shear_flow(1.0, 0.02, 0.15, 1, 1, 0.0, phi, xy_in);
    
    if (ADAPT_STATE_TO_BC) adapt_state_to_bc(phi, bc_field, xy_in);
    
    init_cfield_rde(phi, xy_in, CPLOT, rde, 0);
    if (PLOT_3D) init_zfield_rde(phi, xy_in, ZPLOT, rde, 0);

    if (DOUBLE_MOVIE)
    {
        init_cfield_rde(phi, xy_in, CPLOT_B, rde, 1);
        if (PLOT_3D) init_zfield_rde(phi, xy_in, ZPLOT_B, rde, 1);
    }
    
    if (ADD_TRACERS) for (i=0; i<N_TRACERS; i++)
    {
        tracers[2*i] = XMIN + 0.05 + (XMAX - XMIN - 0.1)*rand()/RAND_MAX;
        tracers[2*i+1] = YMIN + 0.05 + (YMAX - YMIN - 0.1)*rand()/RAND_MAX;
    }
    
    blank();
    glColor3f(0.0, 0.0, 0.0);
    

    glutSwapBuffers();
    
    printf("Drawing wave\n");
    draw_wave_rde(0, phi, xy_in, rde, potential_field, ZPLOT, CPLOT, COLOR_PALETTE, 0, 1.0, 1);
//     draw_billiard();
    if (PRINT_PARAMETERS) print_parameters(rde, xy_in, time, 0, VISCOSITY, noise);
    if (DRAW_COLOR_SCHEME) draw_color_bar_palette(CPLOT, COLORBAR_RANGE, COLOR_PALETTE, 0, 1.0);

    
    glutSwapBuffers();

    sleep(SLEEP1);
//     printf("Saving frame %i\n", i);
    if (MOVIE) for (i=0; i<INITIAL_TIME; i++) save_frame();

    for (i=0; i<=NSTEPS; i++)
    {
        nvid = NVID;
        if (CHANGE_VISCOSITY) 
        {
            viscosity = viscosity_schedule(i);
            viscosity_printed = viscosity;
            printf("Viscosity = %.3lg\n", viscosity); 
            if ((ADJUST_INTSTEP)&&(viscosity > VISCOSITY_MAX))
            {
                nvid = (int)((double)NVID*viscosity/VISCOSITY_MAX);
//                 viscosity = VISCOSITY_MAX;
                intstep = intstep0*VISCOSITY_MAX/viscosity;
                printf("Nvid = %i, intstep = %.3lg\n", nvid, intstep);
            }
        }
        if (CHANGE_RPSLZB) rpslzb = rpslzb_schedule(i);
        if (CHANGE_FLOW_SPEED) flow_speed = flow_speed_schedule(i); 
        else flow_speed = IN_OUT_FLOW_AMP;
        
        if (ROTATE_VIEW) 
        {
            viewpoint_schedule(i - INITIAL_TIME);
            reset_view = 1;
        }
        
        printf("Drawing wave %i\n", i);
        draw_wave_rde(0, phi, xy_in, rde, potential_field, ZPLOT, CPLOT, COLOR_PALETTE, 0, 1.0, 1);
        
//         nvid = (int)((double)NVID*(1.0 + (ACCELERATION_FACTOR - 1.0)*(double)i/(double)NSTEPS));
        /* increase integration step */
//         intstep = intstep0*exp(log(DT_ACCELERATION_FACTOR)*(double)i/(double)NSTEPS);
//         if (intstep > MAX_DT)
//         {
//             nvid *= intstep/MAX_DT;
//             intstep = MAX_DT;
//         }
//         printf("Steps per frame: %i\n", nvid);
//         printf("Integration step %.5lg\n", intstep);
        
        printf("Evolving wave\n");
        for (j=0; j<nvid; j++) evolve_wave(phi, phi_tmp, xy_in, potential_field, vector_potential_field, gfield, rde);

        if (ADAPT_STATE_TO_BC) adapt_state_to_bc(phi, bc_field, xy_in);
        
        if (ADD_TRACERS)
        {
            printf("Evolving tracer particles\n");
            evolve_tracers(phi, tracers, i, 10, 0.1);
//             for (j=0; j<N_TRACERS; j++) 
//                 printf("Tracer %i position (%.2f, %.2f)\n", j, tracers[2*N_TRACERS*i + 2*j], tracers[2*N_TRACERS*i + 2*j + 1]);
            printf("Drawing tracer particles\n");
            draw_tracers(phi, tracers, i, 0, 1.0);
        }
        
        if (ANTISYMMETRIZE_WAVE_FCT) antisymmetrize_wave_function(phi, xy_in);
        
        for (j=0; j<NFIELDS; j++) printf("field[%i] = %.3lg\t", j, phi[j][0]);
        printf("\n");
        
        if (ADD_NOISE == 1) 
        {
//             #pragma omp parallel for private(field,j,k)
            for (field=0; field<NFIELDS; field++)
                for (j=0; j<NX; j++) 
                    for (k=0; k<NY; k++)
                        phi[field][j*NY+k] += sqrintstep*NOISE_INTENSITY*gaussian();
        }
        else if (ADD_NOISE == 2) 
        {
            if (CHANGE_NOISE)
            {
                noise = noise_schedule(i);
//                 #pragma omp parallel for private(field,j,k)
                for (field=0; field<NFIELDS; field++)
                    for (j=NX/2; j<NX; j++) 
                        for (k=0; k<NY; k++)
                            phi[field][j*NY+k] += sqrintstep*noise*gaussian();
            }
            else
            {
//                 #pragma omp parallel for private(field,j,k)
                for (field=0; field<NFIELDS; field++)
                    for (j=NX/2; j<NX; j++) 
                        for (k=0; k<NY; k++)
                            phi[field][j*NY+k] += sqrintstep*NOISE_INTENSITY*gaussian();
            }
        }
        time += nvid*intstep;
        
//         draw_billiard();
        if (PRINT_PARAMETERS) print_parameters(rde, xy_in, time, 0, viscosity_printed, noise);
        if (DRAW_COLOR_SCHEME) draw_color_bar_palette(CPLOT, COLORBAR_RANGE, COLOR_PALETTE, 0, 1.0); 
        
//         print_level(MDEPTH);
//         print_Julia_parameters(i);

        if (!((NO_EXTRA_BUFFER_SWAP)&&(MOVIE))) glutSwapBuffers();
        
//         glutSwapBuffers();
//         save_frame();
        
        /* modify Julia set */
//         set_Julia_parameters(i, phi, xy_in);

	if (MOVIE)
//         if (0)
        {
            printf("Saving frame %i\n", i);
//             if (NO_EXTRA_BUFFER_SWAP) glutSwapBuffers();
            save_frame();
            
            if ((i >= INITIAL_TIME)&&(DOUBLE_MOVIE))
            {
                draw_wave_rde(1, phi, xy_in, rde, potential_field, ZPLOT_B, CPLOT_B, COLOR_PALETTE_B, 0, 1.0, REFRESH_B);
                if (ADD_TRACERS) draw_tracers(phi, tracers, i, 0, 1.0);
//                 draw_billiard();
                if (PRINT_PARAMETERS) print_parameters(rde, xy_in, time, 0, viscosity_printed, noise);
                if (DRAW_COLOR_SCHEME) draw_color_bar_palette(CPLOT_B, COLORBAR_RANGE_B, COLOR_PALETTE_B, 0, 1.0);  
                glutSwapBuffers();
//                 if (NO_EXTRA_BUFFER_SWAP) glutSwapBuffers();
                save_frame_counter(NSTEPS + MID_FRAMES + 1 + counter);
                counter++;
            }
            else if (NO_EXTRA_BUFFER_SWAP) glutSwapBuffers();
            
            /* TEST */
//              if (ADAPT_STATE_TO_BC) adapt_state_to_bc(phi, bc_field, xy_in);

            /* it seems that saving too many files too fast can cause trouble with the file system */
            /* so this is to make a pause from time to time - parameter PAUSE may need adjusting   */
            if (i % PAUSE == PAUSE - 1)
            {
                printf("Making a short pause\n");
                sleep(PSLEEP);
                s = system("mv wave*.tif tif_bz/");
            }
        }
        else printf("Computing frame %i\n", i);

    }
    
    if (MOVIE) 
    {
        if (DOUBLE_MOVIE) 
        {
            draw_wave_rde(0, phi, xy_in, rde, potential_field, ZPLOT, CPLOT, COLOR_PALETTE, 0, 1.0, 1);
            if (ADD_TRACERS) draw_tracers(phi, tracers, NSTEPS, 0, 1.0);
//             draw_billiard();
            if (PRINT_PARAMETERS) print_parameters(rde, xy_in, time, 0, viscosity_printed, noise);
            if (DRAW_COLOR_SCHEME) draw_color_bar_palette(CPLOT, COLORBAR_RANGE, COLOR_PALETTE, 0, 1.0); 
//             if (!NO_EXTRA_BUFFER_SWAP) glutSwapBuffers();
            glutSwapBuffers();
            
            if (!FADE) for (i=0; i<MID_FRAMES; i++) save_frame();
            else for (i=0; i<MID_FRAMES; i++) 
            {
                fade_value = 1.0 - (double)i/(double)MID_FRAMES;
                draw_wave_rde(0, phi, xy_in, rde, potential_field, ZPLOT, CPLOT, COLOR_PALETTE, 1, fade_value, 0);
                if (ADD_TRACERS) draw_tracers(phi, tracers, NSTEPS, 1, fade_value);
//                 draw_billiard();
                if (PRINT_PARAMETERS) print_parameters(rde, xy_in, time, 0, viscosity_printed, noise);
                if (DRAW_COLOR_SCHEME) draw_color_bar_palette(CPLOT, COLORBAR_RANGE, COLOR_PALETTE, 1, fade_value);   
                if (!NO_EXTRA_BUFFER_SWAP) glutSwapBuffers();
                save_frame_counter(NSTEPS + i + 1);
            }
            draw_wave_rde(1, phi, xy_in, rde, potential_field, ZPLOT_B, CPLOT_B, COLOR_PALETTE_B, 0, 1.0, REFRESH_B);
            if (ADD_TRACERS) draw_tracers(phi, tracers, NSTEPS, 0, 1.0);
            if (PRINT_PARAMETERS) print_parameters(rde, xy_in, time, 0, viscosity_printed, noise);
            if (DRAW_COLOR_SCHEME) draw_color_bar_palette(CPLOT_B, COLORBAR_RANGE_B, COLOR_PALETTE_B, 0, 1.0); 
            glutSwapBuffers();
            
            if (!FADE) for (i=0; i<END_FRAMES; i++) save_frame_counter(NSTEPS + MID_FRAMES + 1 + counter + i);
            else for (i=0; i<END_FRAMES; i++) 
            {
                fade_value = 1.0 - (double)i/(double)END_FRAMES;
                draw_wave_rde(1, phi, xy_in, rde, potential_field, ZPLOT_B, CPLOT_B, COLOR_PALETTE_B, 1, fade_value, 0);
                if (ADD_TRACERS) draw_tracers(phi, tracers, NSTEPS, 1, fade_value);
                if (PRINT_PARAMETERS) print_parameters(rde, xy_in, time, 0, viscosity_printed, noise);
                if (DRAW_COLOR_SCHEME) draw_color_bar_palette(CPLOT_B, COLORBAR_RANGE_B, COLOR_PALETTE_B, 1, fade_value);   
                glutSwapBuffers();
                save_frame_counter(NSTEPS + MID_FRAMES + 1 + counter + i);
            }
        }
        else
        {
            if (!FADE) for (i=0; i<END_FRAMES; i++) save_frame_counter(NSTEPS + MID_FRAMES + 1 + counter + i);
            else for (i=0; i<END_FRAMES; i++) 
            {
                fade_value = 1.0 - (double)i/(double)END_FRAMES;
                draw_wave_rde(0, phi, xy_in, rde, potential_field, ZPLOT, CPLOT, COLOR_PALETTE, 1, fade_value, 0);
                if (ADD_TRACERS) draw_tracers(phi, tracers, NSTEPS, 1, fade_value);
                if (DRAW_COLOR_SCHEME) draw_color_bar_palette(CPLOT, COLORBAR_RANGE, COLOR_PALETTE, 1, fade_value); 
                glutSwapBuffers();
                save_frame_counter(NSTEPS + 1 + counter + i);
            }
        }
        
        s = system("mv wave*.tif tif_bz/");
    }

    for (i=0; i<NFIELDS; i++)
    {
        free(phi[i]);
        free(phi_tmp[i]);
    }
    free(xy_in);
    if (ADD_POTENTIAL) free(potential_field);
    else if (ADD_MAGNETIC_FIELD) 
    {
        free(potential_field);
        free(vector_potential_field);
    }
    if (ADD_TRACERS) free(tracers);
    if (ADD_FORCE_FIELD) free(gfield);
    if (ADAPT_STATE_TO_BC) free(bc_field);
    
    printf("Time %.5lg\n", time);

}


void display(void)
{
    time_t rawtime;
    struct tm * timeinfo;

    time(&rawtime);
    timeinfo = localtime(&rawtime);
    
    glPushMatrix();

    blank();
    glutSwapBuffers();
    blank();
    glutSwapBuffers();

    animation();
    sleep(SLEEP2);

    glPopMatrix();

    glutDestroyWindow(glutGetWindow());
    
    printf("Start local time and date: %s", asctime(timeinfo));
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    printf("Current local time and date: %s", asctime(timeinfo));
}


int main(int argc, char** argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(WINWIDTH,WINHEIGHT);
    glutCreateWindow("FitzHugh-Nagumo equation in a planar domain");

    if (PLOT_3D) init_3d(); 
    else init();

    glutDisplayFunc(display);

    glutMainLoop();

    return 0;
}

