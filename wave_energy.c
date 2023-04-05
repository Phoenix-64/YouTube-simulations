/*********************************************************************************/
/*                                                                               */
/*  Animation of wave equation in a planar domain                                */
/*                                                                               */
/*  N. Berglund, december 2012, may  2021                                        */
/*                                                                               */
/*  UPDATE 24/04: distinction between damping and "elasticity" parameters        */
/*  UPDATE 27/04: new billiard shapes, bug in color scheme fixed                 */
/*  UPDATE 28/04: code made more efficient, with help of Marco Mancini           */
/*                                                                               */
/*  Feel free to reuse, but if doing so it would be nice to drop a               */
/*  line to nils.berglund@univ-orleans.fr - Thanks!                              */
/*                                                                               */
/*  compile with                                                                 */
/*  gcc -o wave_billiard wave_billiard.c                                         */
/* -L/usr/X11R6/lib -ltiff -lm -lGL -lGLU -lX11 -lXmu -lglut -O3 -fopenmp        */
/*                                                                               */
/*  OMP acceleration may be more effective after executing                       */
/*  export OMP_NUM_THREADS=2 in the shell before running the program             */
/*                                                                               */
/*  To make a video, set MOVIE to 1 and create subfolder tif_wave                */
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

/* For debugging purposes only */
#define FLOOR 0         /* set to 1 to limit wave amplitude to VMAX */
#define VMAX 10.0       /* max value of wave amplitude */

/*custome added due to error*/
#define ADD_WAVE_PACKET_SOURCES 0       /* set to 1 to add several sources emitting wave packets */
#define WAVE_PACKET_SOURCE_TYPE 0      /* type of wave packet sources */
#define N_WAVE_PACKETS 1     
#define WAVE_PACKET_RADIUS 0.0
#define MAZE_WIDTH 1
#define IOR_TOTAL_TURNS 1
#define MUD  1
#define HEX_NONUNIF_COMPRESSION 0
#define HEX_NONUNIF_COMPRESSION_B 0
#define POLY_ROTATION_ANGLE 0
#define APOLY_B 0
#define XDEP_POLY_ANGLE 0
#define XDEP_POLY_ANGLE_B 0
#define RANDOM_POLY_ANGLE_B 0
#define YMID 0
#define BLACK_TEXT 1



#include "global_pdes.c"        /* constants and global variables */
#include "sub_maze.c"           /* support for generating mazes */
#include "sub_wave.c"           /* common functions for wave_billiard, heat and schrodinger */
#include "wave_common.c"        /* common functions for wave_billiard, wave_comparison, etc */
#include "sub_wave_comp.c"      /* some functions specific to wave_comparison */

double courant2, courantb2;  /* Courant parameters squared */

double compute_energy_x(int i, double *phi[NX], double *psi[NX], short int *xy_in[NX])
/* compute energy in column i */
{
    double energy = 0.0;
    int j;
    
    for (j=0; j<NY/2; j++)
            energy += compute_energy(phi, psi, xy_in, i, j);
    
    return(energy);
}

double logscale_y(double energy)
{
    static double ymid, yscale;
    static int first = 1;
    
    if (first) 
    {
        ymid = 0.5*(YMIN + YMAX);
        yscale = (YMAX - YMIN)*0.5/2.25;
    }
    
    return(ymid + yscale*(1.0 + 0.2*log(energy)));
//     return(ymid + 0.5*(1.0 + 0.2*log(energy)));
}

void draw_wave_energy(double *phi[NX], double *psi[NX], short int *xy_in[NX], double scale, int time)
/* draw the field */
{
    int i, j, iplus, iminus, jplus, jminus;
    double rgb[3], xy[2], x, y, x1, y1, x2, y2, velocity, energy, gradientx2, gradienty2, pos[2], escale;
    double energies[NX], ymid;
    static double dtinverse = ((double)NX)/(COURANT*(XMAX-XMIN)), dx = (XMAX-XMIN)/((double)NX);
    char message[50];

    ymid = 0.5*(YMIN + YMAX);
    
    glBegin(GL_QUADS);
    
//     printf("dtinverse = %.5lg\n", dtinverse);

    for (i=0; i<NX; i++)
        for (j=0; j<NY/2; j++)
        {
            if (((TWOSPEEDS)&&(xy_in[i][j] != 2))||(xy_in[i][j] == 1)) {
                switch (PLOT) {
                    case (P_AMPLITUDE):
                    {
                        /* make wave luminosity larger inside obstacles */
                        if (!(xy_in[i][j])) color_scheme_lum(COLOR_SCHEME, phi[i][j], scale, time, 0.7, rgb);
                        else color_scheme(COLOR_SCHEME, phi[i][j], scale, time, rgb);
                        break;
                    }
                    case (P_ENERGY):
                    {
                        energy = compute_energy(phi, psi, xy_in, i, j);
                        /* adjust energy to color palette */
                        if (COLOR_PALETTE >= COL_TURBO) color_scheme_asym(COLOR_SCHEME, energy, scale, time, rgb);
                        else color_scheme(COLOR_SCHEME, energy, scale, time, rgb);
                        break;
                    }
                    case (P_MIXED):
                    {
                        if (j > NY/2) color_scheme(COLOR_SCHEME, phi[i][j], scale, time, rgb);
                        else color_scheme(COLOR_SCHEME, compute_energy(phi, psi, xy_in, i, j), scale, time, rgb);
                        break;
                    }
                }
                glColor3f(rgb[0], rgb[1], rgb[2]);

                glVertex2i(i, j);
                glVertex2i(i+1, j);
                glVertex2i(i+1, j+1);
                glVertex2i(i, j+1);
            }
        }

    glEnd ();
    
    
    /* compute and plot energies */
    for (i=0; i<NX; i++) energies[i] = compute_energy_x(i, phi, psi, xy_in);
    
    glColor3f(0.0, 0.0, 0.0);
    glBegin(GL_QUADS);
    glVertex2i(0, NY/2);
    glVertex2i(NX, NY/2);
    glVertex2i(NX, NY);
    glVertex2i(0, NY);
    glEnd();
    
    /* log coordinate lines */
    glLineWidth(1);
    glColor3f(1.0, 1.0, 1.0);
    for (i=-2; i<3; i++)
    {
        energy = pow(10.0, (double)i);
        y = logscale_y(energy);
        glBegin(GL_LINE_STRIP);
        x = XMIN;
        xy_to_pos(x, y, pos);
        glVertex2d(pos[0], pos[1]);
        x = XMAX;
        xy_to_pos(x, y, pos);
        glVertex2d(pos[0], pos[1]);
        glEnd();
    }
    glColor3f(0.5, 0.5, 0.5);
    for (i=-2; i<3; i++)
    {
        for (j=2; j<10; j++)
        {
            energy = (double)j*pow(10.0, (double)i);
            y = logscale_y(energy);
            glBegin(GL_LINE_STRIP);
            x = XMIN;
            xy_to_pos(x, y, pos);
            glVertex2d(pos[0], pos[1]);
            x = XMAX;
            xy_to_pos(x, y, pos);
            glVertex2d(pos[0], pos[1]);
            glEnd();
        }
    }
    
    erase_area_hsl(XMAX - 0.4, YMAX - 0.1, 0.35, 0.07, 0.0, 1.0, 0.0); 
    erase_area_hsl(XMAX - 0.4, YMAX - 0.2, 0.35, 0.07, 0.0, 1.0, 0.0); 
    
    sprintf(message, "Energy (log scale)");
    glColor3f(0.0, 0.5, 1.0);
    xy_to_pos(XMAX - 0.7, YMAX - 0.13, pos);
    write_text(pos[0], pos[1], message);
    sprintf(message, "Energy (linear scale)");
    glColor3f(1.0, 0.0, 0.0);
    xy_to_pos(XMAX - 0.7, YMAX - 0.23, pos);
    write_text(pos[0], pos[1], message);
    
    /* log of energy */
    glLineWidth(3);
    glColor3f(0.0, 0.5, 1.0);
    glBegin(GL_LINE_STRIP);
    for (i=0; i<NX; i++)
    {
        x = XMIN + ((double)i)*(XMAX-XMIN)/((double)NX);
        y = logscale_y(energies[i]);
        if (y < ymid) y = ymid;
        xy_to_pos(x, y, pos);
        glVertex2d(pos[0], pos[1]);
    }
    glEnd();
    
    /* y axis labels */
    for (i=-2; i<3; i++)
    {
        y = logscale_y(pow(10.0, (double)i));
        erase_area_hsl(XMIN + 0.06, y + 0.025, 0.12, 0.02, 0.0, 1.0, 0.0); 
        sprintf(message, "%d dB", (i-2)*10);
        xy_to_pos(XMIN + 0.02, y + 0.01, pos);
        glColor3f(0.7, 0.7, 0.7);
        write_text_fixedwidth(pos[0], pos[1], message);
    }

    /* energy */
    glColor3f(1.0, 0.0, 0.0);
    escale = 0.01;
    glBegin(GL_LINE_STRIP);
    for (i=0; i<NX; i++)
    {
        x = XMIN + ((double)i)*(XMAX-XMIN)/((double)NX);
        y = ymid + escale*energies[i];
        xy_to_pos(x, y, pos);
        glVertex2d(pos[0], pos[1]);
    }
    glEnd();
    

    /* draw horizontal mid line */
    glColor3f(1.0, 1.0, 1.0);
    glBegin(GL_LINE_STRIP);
    xy_to_pos(XMIN, 0.5*(YMIN+YMAX), pos);
    glVertex2d(pos[0], pos[1]);
    xy_to_pos(XMAX, 0.5*(YMIN+YMAX), pos);
    glVertex2d(pos[0], pos[1]);
    glEnd();
}

/*********************/
/* animation part    */
/*********************/

void evolve_wave_half_old(double *phi_in[NX], double *psi_in[NX], double *phi_out[NX], double *psi_out[NX], 
                      short int *xy_in[NX])
/* time step of field evolution */
/* phi is value of field at time t, psi at time t-1 */
{
    int i, j, iplus, iminus, jplus, jminus, jmid = NY/2;
    double delta, x, y, c, cc, gamma;
    static long time = 0;
    
    time++;

    #pragma omp parallel for private(i,j,iplus,iminus,jplus,jminus,delta,x,y,c,cc,gamma)
    for (i=0; i<NX; i++){
        for (j=0; j<NY/2; j++){
            if (xy_in[i][j])
            {
                c = COURANT;
                cc = courant2;
                gamma = GAMMA;
            }
            else if (TWOSPEEDS)
            {
                c = COURANTB;
                cc = courantb2;
                gamma = GAMMAB;
            }

            if (((TWOSPEEDS)&&(xy_in[i][j] != 2))||(xy_in[i][j] == 1)){
                /* discretized Laplacian for various boundary conditions */
                if ((B_COND == BC_DIRICHLET)||(B_COND == BC_ABSORBING)||(B_COND == BC_ABS_REFLECT))
                {
                    iplus = (i+1);   if (iplus == NX) iplus = NX-1;
                    iminus = (i-1);  if (iminus == -1) iminus = 0;
                    jplus = (j+1);   if (jplus == jmid) jplus = jmid-1;
                    jminus = (j-1);  if (jminus == -1) jminus = 0;
                }
                else if (B_COND == BC_PERIODIC)
                {
                    iplus = (i+1) % NX;
                    iminus = (i-1) % NX;    if (iminus < 0) iminus += NX;
                    jplus = (j+1) % jmid;
                    jminus = (j-1) % jmid;  if (jminus < 0) jminus += jmid;
                }
                else if (B_COND == BC_VPER_HABS)
                {
                    iplus = (i+1);   if (iplus == NX) iplus = NX-1;
                    iminus = (i-1);  if (iminus == -1) iminus = 0;
                    jplus = (j+1);   if (jplus >= jmid) jplus -= jmid;
                    jminus = (j-1);  if (jminus < 0) jminus += jmid;
                }
                
                /* imposing linear wave on top and bottom by making Laplacian 1d */
                if (OSCILLATE_TOPBOT)
                {
                    if (j == NY-1) jminus = NY-1;
                    else if (j == 0) jplus = 0;
                }
                
                delta = phi_in[iplus][j] + phi_in[iminus][j] + phi_in[i][jplus] + phi_in[i][jminus] - 4.0*phi_in[i][j];

                x = phi_in[i][j];
		y = psi_in[i][j];

                /* evolve phi */
                if ((B_COND == BC_PERIODIC)||(B_COND == BC_DIRICHLET)) 
                    phi_out[i][j] = -y + 2*x + cc*delta - KAPPA*x - gamma*(x-y);
                else if ((B_COND == BC_ABSORBING)||(B_COND == BC_ABS_REFLECT))
                {
                    if ((i>0)&&(i<NX-1)&&(j>0)&&(j<NY-1))
                        phi_out[i][j] = -y + 2*x + cc*delta - KAPPA*x - gamma*(x-y);
                
                    /* upper border */
                    else if (j==NY-1) 
                        phi_out[i][j] = x - c*(x - phi_in[i][NY-2]) - KAPPA_TOPBOT*x - GAMMA_TOPBOT*(x-y);
                    
                    /* lower border */
                    else if (j==0) 
                        phi_out[i][j] = x - c*(x - phi_in[i][1]) - KAPPA_TOPBOT*x - GAMMA_TOPBOT*(x-y);
                
                    /* right border */
                    if (i==NX-1) 
                        phi_out[i][j] = x - c*(x - phi_in[NX-2][j]) - KAPPA_SIDES*x - GAMMA_SIDES*(x-y);
                    
                    /* left border */
                    else if (i==0) 
                        phi_out[i][j] = x - c*(x - phi_in[1][j]) - KAPPA_SIDES*x - GAMMA_SIDES*(x-y);
                }
                else if (B_COND == BC_VPER_HABS)
                {
                    if ((i>0)&&(i<NX-1))
                        phi_out[i][j] = -y + 2*x + cc*delta - KAPPA*x - gamma*(x-y);
                
                    /* right border */
                    else if (i==NX-1) 
                        phi_out[i][j] = x - c*(x - phi_in[NX-2][j]) - KAPPA_SIDES*x - GAMMA_SIDES*(x-y);
                    
                    /* left border */
                    else if (i==0) 
                        phi_out[i][j] = x - c*(x - phi_in[1][j]) - KAPPA_SIDES*x - GAMMA_SIDES*(x-y);
                }
                
                /* add oscillating boundary condition on the left */
                if ((i == 0)&&(OSCILLATE_LEFT)) phi_out[i][j] = AMPLITUDE*cos((double)time*OMEGA);
                
                psi_out[i][j] = x;

                if (FLOOR)
                {
                    if (phi_out[i][j] > VMAX) phi_out[i][j] = VMAX;
                    if (phi_out[i][j] < -VMAX) phi_out[i][j] = -VMAX;
                    if (psi_out[i][j] > VMAX) psi_out[i][j] = VMAX;
                    if (psi_out[i][j] < -VMAX) psi_out[i][j] = -VMAX;
                }
            }
        }
    }
//     printf("phi(0,0) = %.3lg, psi(0,0) = %.3lg\n", phi[NX/2][NY/2], psi[NX/2][NY/2]);
}


void evolve_wave_half(double *phi_in[NX], double *psi_in[NX], double *phi_out[NX],
                      short int *xy_in[NX])
/* time step of field evolution */
/* phi is value of field at time t, psi at time t-1 */
{
    int i, j, iplus, iminus, jplus, jminus, jmid = NY/2;
    double delta, x, y, c, cc, gamma;
    static long time = 0;
    static double tc[NX][NY/2], tcc[NX][NY/2], tgamma[NX][NY/2];
    static short int first = 1;
    
    time++;
    
    /* initialize tables with wave speeds and dissipation */
    if (first)
    {
        for (i=0; i<NX; i++){
            for (j=0; j<jmid; j++){
                if (xy_in[i][j])
                {
                    tc[i][j] = COURANT;
                    tcc[i][j] = courant2;
                    tgamma[i][j] = GAMMA;
                }
                else if (TWOSPEEDS)
                {
                    tc[i][j] = COURANTB;
                    tcc[i][j] = courantb2;
                    tgamma[i][j] = GAMMAB;
                }
            }
        }
        first = 0;
    }

    #pragma omp parallel for private(i,j,iplus,iminus,jplus,jminus,delta,x,y,c,cc,gamma)
    /* evolution in the bulk */
    for (i=1; i<NX-1; i++){
        for (j=1; j<jmid-1; j++){
            if ((TWOSPEEDS)||(xy_in[i][j] != 0)){
                x = phi_in[i][j];
		y = psi_in[i][j];
                
                /* discretized Laplacian */
                delta = phi_in[i+1][j] + phi_in[i-1][j] + phi_in[i][j+1] + phi_in[i][j-1] - 4.0*x;

                /* evolve phi */
                phi_out[i][j] = -y + 2*x + tcc[i][j]*delta - KAPPA*x - tgamma[i][j]*(x-y);
            }
        }
    }
    
    /* left boundary */
    if (OSCILLATE_LEFT) for (j=1; j<jmid; j++) phi_out[0][j] = AMPLITUDE*cos((double)time*OMEGA);
    else for (j=1; j<jmid-1; j++){
        if ((TWOSPEEDS)||(xy_in[0][j] != 0)){
            x = phi_in[0][j];
            y = psi_in[0][j];
                    
            switch (B_COND) {
                case (BC_DIRICHLET):
                {
                    delta = phi_in[1][j] + phi_in[0][j+1] + phi_in[0][j-1] - 3.0*x;
                    phi_out[0][j] = -y + 2*x + tcc[0][j]*delta - KAPPA*x - tgamma[0][j]*(x-y);
                    break;
                }
                case (BC_PERIODIC):
                {
                    delta = phi_in[1][j] + phi_in[NX-1][j] + phi_in[0][j+1] + phi_in[0][j-1] - 4.0*x;
                    phi_out[0][j] = -y + 2*x + tcc[0][j]*delta - KAPPA*x - tgamma[0][j]*(x-y);
                    break;
                }
                case (BC_ABSORBING):
                {
                    delta = phi_in[1][j] + phi_in[0][j+1] + phi_in[0][j-1] - 3.0*x;
                    phi_out[0][j] = x - tc[0][j]*(x - phi_in[1][j]) - KAPPA_SIDES*x - GAMMA_SIDES*(x-y);
                    break;
                }
                case (BC_VPER_HABS):
                {
                    delta = phi_in[1][j] + phi_in[0][j+1] + phi_in[0][j-1] - 3.0*x;
                    phi_out[0][j] = x - tc[0][j]*(x - phi_in[1][j]) - KAPPA_SIDES*x - GAMMA_SIDES*(x-y);
                    break;
                }
            }
        }
    }
    
    /* right boundary */
    for (j=1; j<jmid-1; j++){
        if ((TWOSPEEDS)||(xy_in[NX-1][j] != 0)){
            x = phi_in[NX-1][j];
            y = psi_in[NX-1][j];
                    
            switch (B_COND) {
                case (BC_DIRICHLET):
                {
                    delta = phi_in[NX-2][j] + phi_in[NX-1][j+1] + phi_in[NX-1][j-1] - 3.0*x;
                    phi_out[NX-1][j] = -y + 2*x + tcc[NX-1][j]*delta - KAPPA*x - tgamma[NX-1][j]*(x-y);
                    break;
                }
                case (BC_PERIODIC):
                {
                    delta = phi_in[NX-2][j] + phi_in[0][j] + phi_in[NX-1][j+1] + phi_in[NX-1][j-1] - 4.0*x;
                    phi_out[NX-1][j] = -y + 2*x + tcc[NX-1][j]*delta - KAPPA*x - tgamma[NX-1][j]*(x-y);
                    break;
                }
                case (BC_ABSORBING):
                {
                    delta = phi_in[NX-2][j] + phi_in[NX-1][j+1] + phi_in[NX-1][j-1] - 3.0*x;
                    phi_out[NX-1][j] = x - tc[NX-1][j]*(x - phi_in[NX-2][j]) - KAPPA_SIDES*x - GAMMA_SIDES*(x-y);
                    break;
                }
                case (BC_VPER_HABS):
                {
                    delta = phi_in[NX-2][j] + phi_in[NX-1][j+1] + phi_in[NX-1][j-1] - 3.0*x;
                    phi_out[NX-1][j] = x - tc[NX-1][j]*(x - phi_in[NX-2][j]) - KAPPA_SIDES*x - GAMMA_SIDES*(x-y);
                    break;
                }
            }
        }
    }
    
    /* top boundary */
    for (i=0; i<NX; i++){
        if ((TWOSPEEDS)||(xy_in[i][jmid-1] != 0)){
            x = phi_in[i][jmid-1];
            y = psi_in[i][jmid-1];
                    
            switch (B_COND) {
                case (BC_DIRICHLET):
                {
                    iplus = i+1;   if (iplus == NX) iplus = NX-1;
                    iminus = i-1;  if (iminus == -1) iminus = 0;
                    
                    delta = phi_in[iplus][jmid-1] + phi_in[iminus][jmid-1] + phi_in[i][jmid-2] - 3.0*x;
                    phi_out[i][jmid-1] = -y + 2*x + tcc[i][jmid-1]*delta - KAPPA*x - tgamma[i][jmid-1]*(x-y);
                    break;
                }
                case (BC_PERIODIC):
                {
                    iplus = (i+1) % NX;
                    iminus = (i-1) % NX;    if (iminus < 0) iminus += NX;
                    
                    delta = phi_in[iplus][jmid-1] + phi_in[iminus][jmid-1] + phi_in[i][jmid-2] + phi_in[i][0] - 4.0*x;
                    phi_out[i][jmid-1] = -y + 2*x + tcc[i][jmid-1]*delta - KAPPA*x - tgamma[i][jmid-1]*(x-y);
                    break;
                }
                case (BC_ABSORBING):
                {
                    iplus = (i+1);   if (iplus == NX) iplus = NX-1;
                    iminus = (i-1);  if (iminus == -1) iminus = 0;
                    
                    delta = phi_in[iplus][jmid-1] + phi_in[iminus][jmid-1] + phi_in[i][jmid-2] - 3.0*x;
                    phi_out[i][jmid-1] = x - tc[i][jmid-1]*(x - phi_in[i][jmid-2]) - KAPPA_TOPBOT*x - GAMMA_TOPBOT*(x-y);
                    break;
                }
                case (BC_VPER_HABS):
                {
                    iplus = (i+1);   if (iplus == NX) iplus = NX-1;
                    iminus = (i-1);  if (iminus == -1) iminus = 0;

                    delta = phi_in[iplus][jmid-1] + phi_in[iminus][jmid-1] + phi_in[i][jmid-2] + phi_in[i][0] - 4.0*x;
                    if (i==0) phi_out[0][jmid-1] = x - tc[0][jmid-1]*(x - phi_in[1][jmid-1]) - KAPPA_SIDES*x - GAMMA_SIDES*(x-y);
                    else phi_out[i][jmid-1] = -y + 2*x + tcc[i][jmid-1]*delta - KAPPA*x - tgamma[i][jmid-1]*(x-y);
//                      delta = phi_in[1][j] + phi_in[0][j+1] + phi_in[0][j-1] - 3.0*x;
//                     phi_out[0][j] = x - tc[0][j]*(x - phi_in[1][j]) - KAPPA_SIDES*x - GAMMA_SIDES*(x-y);
                   break;
                }
            }
        }
    }
    
    /* bottom boundary */
    for (i=0; i<NX; i++){
        if ((TWOSPEEDS)||(xy_in[i][0] != 0)){
            x = phi_in[i][0];
            y = psi_in[i][0];
                    
            switch (B_COND) {
                case (BC_DIRICHLET):
                {
                    iplus = i+1;   if (iplus == NX) iplus = NX-1;
                    iminus = i-1;  if (iminus == -1) iminus = 0;
                    
                    delta = phi_in[iplus][0] + phi_in[iminus][0] + phi_in[i][1] - 3.0*x;
                    phi_out[i][0] = -y + 2*x + tcc[i][0]*delta - KAPPA*x - tgamma[i][0]*(x-y);
                    break;
                }
                case (BC_PERIODIC):
                {
                    iplus = (i+1) % NX;
                    iminus = (i-1) % NX;    if (iminus < 0) iminus += NX;
                    
                    delta = phi_in[iplus][0] + phi_in[iminus][0] + phi_in[i][1] + phi_in[i][jmid-1] - 4.0*x;
                    phi_out[i][0] = -y + 2*x + tcc[i][0]*delta - KAPPA*x - tgamma[i][0]*(x-y);
                    break;
                }
                case (BC_ABSORBING):
                {
                    iplus = (i+1);   if (iplus == NX) iplus = NX-1;
                    iminus = (i-1);  if (iminus == -1) iminus = 0;
                    
                    delta = phi_in[iplus][0] + phi_in[iminus][0] + phi_in[i][1] - 3.0*x;
                    phi_out[i][0] = x - tc[i][0]*(x - phi_in[i][1]) - KAPPA_TOPBOT*x - GAMMA_TOPBOT*(x-y);
                    break;
                }
                case (BC_VPER_HABS):
                {
                    iplus = (i+1);   if (iplus == NX) iplus = NX-1;
                    iminus = (i-1);  if (iminus == -1) iminus = 0;

                    delta = phi_in[iplus][0] + phi_in[iminus][0] + phi_in[i][1] + phi_in[i][jmid-1] - 4.0*x;
                    if (i==0) phi_out[0][0] = x - tc[0][0]*(x - phi_in[1][0]) - KAPPA_SIDES*x - GAMMA_SIDES*(x-y);
                    else phi_out[i][0] = -y + 2*x + tcc[i][0]*delta - KAPPA*x - tgamma[i][0]*(x-y);
                    break;
                }
            }
        }
    }
    
    /* add oscillating boundary condition on the left corners */
    if ((i == 0)&&(OSCILLATE_LEFT))
    {
        phi_out[i][0] = AMPLITUDE*cos((double)time*OMEGA);
        phi_out[i][jmid-1] = AMPLITUDE*cos((double)time*OMEGA);
    }
    
    /* for debugging purposes/if there is a risk of blow-up */
    if (FLOOR) for (i=0; i<NX; i++){
        for (j=0; j<jmid; j++){
            if (xy_in[i][j] != 0) 
            {
                if (phi_out[i][j] > VMAX) phi_out[i][j] = VMAX;
                if (phi_out[i][j] < -VMAX) phi_out[i][j] = -VMAX;
            }
        }
    }
//     printf("phi(0,0) = %.3lg, psi(0,0) = %.3lg\n", phi[NX/2][NY/2], psi[NX/2][NY/2]);
}


void evolve_wave(double *phi[NX], double *psi[NX], double *tmp[NX], short int *xy_in[NX])
/* time step of field evolution */
/* phi is value of field at time t, psi at time t-1 */
{
//     evolve_wave_half_old(phi, psi, phi_tmp, psi_tmp, xy_in);
//     evolve_wave_half_old(phi_tmp, psi_tmp, phi, psi, xy_in);
    evolve_wave_half(phi, psi, tmp, xy_in);
    evolve_wave_half(tmp, phi, psi, xy_in);
    evolve_wave_half(psi, tmp, phi, xy_in);
}



void animation()
{
    double time, scale, energies[6], top_energy, bottom_energy;
    double *phi[NX], *psi[NX], *tmp[NX];
    short int *xy_in[NX];
    int i, j, s;

    /* Since NX and NY are big, it seemed wiser to use some memory allocation here */
    for (i=0; i<NX; i++)
    {
        phi[i] = (double *)malloc(NY*sizeof(double));
        psi[i] = (double *)malloc(NY*sizeof(double));
        tmp[i] = (double *)malloc(NY*sizeof(double));
        xy_in[i] = (short int *)malloc(NY*sizeof(short int));
    }
    
    /* initialise positions and radii of circles */
    printf("initializing circle configuration\n");
    if ((B_DOMAIN == D_CIRCLES)||(B_DOMAIN_B == D_CIRCLES)) init_circle_config_energy(circles);
    else if (B_DOMAIN == D_POLYGONS) init_polygon_config(polygons);
    

    courant2 = COURANT*COURANT;
    courantb2 = COURANTB*COURANTB;

    /* initialize wave with a drop at one point, zero elsewhere */
//     init_wave_flat_comp(phi, psi, xy_in);
    int_planar_wave_comp(XMIN + 0.015, 0.0, phi, psi, xy_in);
//     int_planar_wave_comp(XMIN + 0.5, 0.0, phi, psi, xy_in);
    printf("initializing wave\n");
//     int_planar_wave_comp(XMIN + 0.1, 0.0, phi, psi, xy_in);
//     int_planar_wave_comp(XMIN + 1.0, 0.0, phi, psi, xy_in);
//     init_wave(-1.5, 0.0, phi, psi, xy_in);
//     init_wave(0.0, 0.0, phi, psi, xy_in);

    /* add a drop at another point */
//     add_drop_to_wave(1.0, 0.7, 0.0, phi, psi);
//     add_drop_to_wave(1.0, -0.7, 0.0, phi, psi);
//     add_drop_to_wave(1.0, 0.0, -0.7, phi, psi);


    blank();
    glColor3f(0.0, 0.0, 0.0);
    printf("drawing wave\n");
    draw_wave_energy(phi, psi, xy_in, 1.0, 0);

    printf("drawing billiard\n");
    draw_billiard_half(B_DOMAIN, CIRCLE_PATTERN, 0);

    glutSwapBuffers();



    sleep(SLEEP1);

    for (i=0; i<=INITIAL_TIME + NSTEPS; i++)
    {
	//printf("%d\n",i);
        /* compute the variance of the field to adjust color scheme */
        /* the color depends on the field divided by sqrt(1 + variance) */
        if (SCALE)
        {
            scale = sqrt(1.0 + compute_variance(phi,psi, xy_in));
//             printf("Scaling factor: %5lg\n", scale);
        }
        else scale = 1.0;

        draw_wave_energy(phi, psi, xy_in, scale, i);
        
        draw_billiard_half(B_DOMAIN, CIRCLE_PATTERN, 0);


        
        for (j=0; j<NVID; j++) 
        {
            evolve_wave(phi, psi, tmp, xy_in);
//             if (i % 10 == 9) oscillate_linear_wave(0.2*scale, 0.15*(double)(i*NVID + j), -1.5, YMIN, -1.5, YMAX, phi, psi);
        }
        
        glutSwapBuffers();

	if (MOVIE)
        {
            if (i >= INITIAL_TIME) save_frame();
            else printf("Initial phase time %i of %i\n", i, INITIAL_TIME);

            /* it seems that saving too many files too fast can cause trouble with the file system */
            /* so this is to make a pause from time to time - parameter PAUSE may need adjusting   */
            if (i % PAUSE == PAUSE - 1)
            {
                printf("Making a short pause\n");
                sleep(PSLEEP);
                s = system("mv wave*.tif tif_wave/");
            }
        }

    }

    if (MOVIE) 
    {
        for (i=0; i<END_FRAMES; i++) save_frame();
        s = system("mv wave*.tif tif_wave/");
    }
    for (i=0; i<NX; i++)
    {
        free(phi[i]);
        free(psi[i]);
        free(tmp[i]);
        free(xy_in[i]);
    }

}


void display(void)
{
    glPushMatrix();

    blank();
    glutSwapBuffers();
    blank();
    glutSwapBuffers();

    animation();
    sleep(SLEEP2);

    glPopMatrix();

    glutDestroyWindow(glutGetWindow());

}


int main(int argc, char** argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(WINWIDTH,WINHEIGHT);
    glutCreateWindow("Wave equation in a planar domain");

    init();

    glutDisplayFunc(display);

    glutMainLoop();

    return 0;
}
