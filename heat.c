/*********************************************************************************/
/*                                                                               */
/*  Animation of heat equation in a planar domain                                */
/*                                                                               */
/*  N. Berglund, May 2021                                                        */
/*                                                                               */
/*  Feel free to reuse, but if doing so it would be nice to drop a               */
/*  line to nils.berglund@univ-orleans.fr - Thanks!                              */
/*                                                                               */
/*  compile with                                                                 */
/*  gcc -o heat heat.c                                                           */
/* -L/usr/X11R6/lib -ltiff -lm -lGL -lGLU -lX11 -lXmu -lglut -O3 -fopenmp        */
/*                                                                               */
/*  To make a video, set MOVIE to 1 and create subfolder tif_heat                */
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

/* added missing defines*/
#define DRAW_BILLIARD 0
#define DT 0.00000025
#define VISCOSITY 2.0
#define SPEED 0.0       /* speed of drift to the right */
#define F_GRADIENT 0
#define F_INTENSITY 0
#define FIELD_REP 0
#define DRAW_FIELD_LINES 0
#define FIELD_LINE_FACTOR 0
#define N_FIELD_LINES 0
#define FIELD_LINE_WIDTH 1
#define T_IN 0.0        /* inside temperature */
#define T_OUT 2.0       /* outside temperature */
#define ADD_WAVE_PACKET_SOURCES 1       /* set to 1 to add several sources emitting wave packets */
#define WAVE_PACKET_SOURCE_TYPE 1       /* type of wave packet sources */
#define N_WAVE_PACKETS 15     
#define WAVE_PACKET_RADIUS 1.0
#define MAZE_WIDTH 1
#define IOR_TOTAL_TURNS 1


#include "global_pdes.c"
#include "sub_maze.c"
#include "sub_wave.c"


double courant2;  /* Courant parameter squared */
double dx2;       /* spatial step size squared */
double intstep;   /* integration step */
double intstep1;  /* integration step used in absorbing boundary conditions */



void init_gaussian(double x, double y, double mean, double amplitude, double scalex, 
                   double *phi[NX], short int * xy_in[NX])
/* initialise field with gaussian at position (x,y) */
{
    int i, j, in;
    double xy[2], dist2, module, phase, scale2;    

    scale2 = scalex*scalex;
    printf("Initialising field\n");
    for (i=0; i<NX; i++)
        for (j=0; j<NY; j++)
        {
            ij_to_xy(i, j, xy);
	    xy_in[i][j] = xy_in_billiard(xy[0],xy[1]);

            in = xy_in[i][j];
            if (in == 1)
            {
                dist2 = (xy[0]-x)*(xy[0]-x) + (xy[1]-y)*(xy[1]-y);
                module = amplitude*exp(-dist2/scale2);
                if (module < 1.0e-15) module = 1.0e-15;

                phi[i][j] = mean + module/scalex;
            }   /* boundary temperatures */
            else if (in >= 2) phi[i][j] = T_IN*pow(0.75, (double)(in-2));
//             else if (in >= 2) phi[i][j] = T_IN*pow(1.0 - 0.5*(double)(in-2), (double)(in-2));
//             else if (in >= 2) phi[i][j] = T_IN*(1.0 - (double)(in-2)/((double)MDEPTH))*(1.0 - (double)(in-2)/((double)MDEPTH));
            else phi[i][j] = T_OUT;
        }
}

void init_julia_set(double *phi[NX], short int * xy_in[NX])
/* change Julia set boundary condition */
{
    int i, j, in;
    double xy[2], dist2, module, phase, scale2;    

//     printf("Changing Julia set\n");
    for (i=0; i<NX; i++)
        for (j=0; j<NY; j++)
        {
            ij_to_xy(i, j, xy);
	    xy_in[i][j] = xy_in_billiard(xy[0],xy[1]);

            in = xy_in[i][j];
            if (in >= 2) phi[i][j] = T_IN;
        }
}


/*********************/
/* animation part    */
/*********************/


void compute_gradient(double *phi[NX], double *nablax[NX], double *nablay[NX])
/* compute the gradient of the field */
{
    int i, j, iplus, iminus, jplus, jminus; 
    double dx;
    
    dx = (XMAX-XMIN)/((double)NX);
    
    for (i=0; i<NX; i++)
        for (j=0; j<NY; j++)
        {
            iplus = i+1;  if (iplus == NX) iplus = NX-1;
            iminus = i-1; if (iminus == -1) iminus = 0;
            jplus = j+1;  if (jplus == NX) jplus = NY-1;
            jminus = j-1; if (jminus == -1) jminus = 0;
            nablax[i][j] = (phi[iplus][j] - phi[iminus][j])/dx;
            nablay[i][j] = (phi[i][jplus] - phi[i][jminus])/dx;
        }
}

void draw_field_line(double x, double y, short int *xy_in[NX], double *nablax[NX], 
                     double *nablay[NX], double delta, int nsteps)
/* draw a field line of the gradient, starting in (x,y) */
{
    double x1, y1, x2, y2, pos[2], nabx, naby, norm2, norm;
    int i = 0, ij[2], cont = 1;
    
    glColor3f(1.0, 1.0, 1.0);
//     glColor3f(0.0, 0.0, 0.0);
    glLineWidth(FIELD_LINE_WIDTH);
    x1 = x;
    y1 = y;
    
//     printf("Drawing field line \n");

    glEnable(GL_LINE_SMOOTH);
    glBegin(GL_LINE_STRIP);
    xy_to_pos(x1, y1, pos);
    glVertex2d(pos[0], pos[1]);
    
    i = 0;
    while ((cont)&&(i < nsteps))
    {
        xy_to_ij(x1, y1, ij);
        
        if (ij[0] < 0) ij[0] = 0;
        if (ij[0] > NX-1) ij[0] = NX-1;
        if (ij[1] < 0) ij[1] = 0;
        if (ij[1] > NY-1) ij[1] = NY-1;
        
        nabx = nablax[ij[0]][ij[1]];
        naby = nablay[ij[0]][ij[1]];
        
        norm2 = nabx*nabx + naby*naby;
        
        if (norm2 > 1.0e-14)
        {
            /* avoid too large step size */
            if (norm2 < 1.0e-9) norm2 = 1.0e-9;
            norm = sqrt(norm2);
            x1 = x1 + delta*nabx/norm;
            y1 = y1 + delta*naby/norm;
        }
        else cont = 0;
        
        if (!xy_in[ij[0]][ij[1]]) cont = 0;
        
        /* stop if the boundary is hit */
//         if (xy_in[ij[0]][ij[1]] != 1) cont = 0;
        
//         printf("x1 = %.3lg \t y1 = %.3lg \n", x1, y1);
                
        xy_to_pos(x1, y1, pos);
        glVertex2d(pos[0], pos[1]);
        
        i++;
    }
    glEnd();
}

void draw_wave(double *phi[NX], short int *xy_in[NX], double scale, int time)
/* draw the field */
{
    int i, j, iplus, iminus, jplus, jminus, ij[2], counter = 0;
    static int first = 1;
    double rgb[3], xy[2], x1, y1, x2, y2, dx, value, angle, dangle, intens, deltaintens, sum = 0.0;
    double *nablax[NX], *nablay[NX];
    static double linex[N_FIELD_LINES*FIELD_LINE_FACTOR], liney[N_FIELD_LINES*FIELD_LINE_FACTOR], distance[N_FIELD_LINES*FIELD_LINE_FACTOR], integral[N_FIELD_LINES*FIELD_LINE_FACTOR + 1];

    for (i=0; i<NX; i++) 
    {
        nablax[i] = (double *)malloc(NY*sizeof(double));
        nablay[i] = (double *)malloc(NY*sizeof(double));
    }
    
    /* compute the gradient */
    compute_gradient(phi, nablax, nablay);
    
    /* compute the position of origins of field lines */
    if ((first)&&(DRAW_FIELD_LINES))
    {
        first = 0;
        
        printf("computing linex\n");
        
//         x1 = LAMBDA + MU*1.01;
//         y1 = 1.0;
        x1 = 0.99*LAMBDA;
        y1 = 0.0;
        linex[0] = x1;
        liney[0] = y1;
        dangle = DPI/((double)(N_FIELD_LINES*FIELD_LINE_FACTOR));
            
        for (i = 1; i < N_FIELD_LINES*FIELD_LINE_FACTOR; i++)
        {
            angle = (double)i*dangle;
//             x2 = LAMBDA + MU*1.01*cos(angle);
//             y2 = 0.5 + MU*1.01*sin(angle);
            x2 = 0.99*LAMBDA*cos(angle);
            y2 = 0.99*LAMBDA*sin(angle);
            linex[i] = x2;
            liney[i] = y2;
            distance[i-1] = module2(x2-x1,y2-y1);
            x1 = x2;
            y1 = y2;
        }
        distance[N_FIELD_LINES*FIELD_LINE_FACTOR - 1] = module2(x2- 0.99*LAMBDA,y2);
//         distance[N_FIELD_LINES*FIELD_LINE_FACTOR - 1] = module2(x2-LAMBDA,y2-0.5);
    }

    dx = (XMAX-XMIN)/((double)NX);
    glBegin(GL_QUADS);

    for (i=0; i<NX; i++)
        for (j=0; j<NY; j++)
        {
            if (FIELD_REP == F_INTENSITY) value = phi[i][j];
            else if (FIELD_REP == F_GRADIENT)
            {
                value = module2(nablax[i][j], nablay[i][j]);
            }
            
            if (xy_in[i][j] == 1) 
            {
                color_scheme(COLOR_SCHEME, value, scale, time, rgb);
                glColor3f(rgb[0], rgb[1], rgb[2]);
            }
            else glColor3f(0.0, 0.0, 0.0);

            glVertex2i(i, j);
            glVertex2i(i+1, j);
            glVertex2i(i+1, j+1);
            glVertex2i(i, j+1);
        }
    glEnd ();
        
    /* draw a field line */
    if (DRAW_FIELD_LINES)
    {
        /* compute gradient norm along boundary and its integral */
        for (i = 0; i < N_FIELD_LINES*FIELD_LINE_FACTOR; i++)
        {
            xy_to_ij(linex[i], liney[i], ij);
            intens = module2(nablax[ij[0]][ij[1]], nablay[ij[0]][ij[1]])*distance[i];
            if (i > 0) integral[i] = integral[i-1] + intens;
            else integral[i] = intens;
        }
        deltaintens = integral[N_FIELD_LINES*FIELD_LINE_FACTOR-1]/((double)N_FIELD_LINES);
        
//         printf("delta = %.5lg\n", deltaintens);
        
        i = 0;
        draw_field_line(linex[0], liney[0], xy_in, nablax, nablay, 0.00002, 100000);
        for (j = 1; j < N_FIELD_LINES+1; j++)
        {
            while ((integral[i] <= j*deltaintens)&&(i < N_FIELD_LINES*FIELD_LINE_FACTOR)) i++; 
            draw_field_line(linex[i], liney[i], xy_in, nablax, nablay, 0.00002, 100000);
            counter++;
        }
        printf("%i lines\n", counter);
    }

    
    for (i=0; i<NX; i++)
    {
        free(nablax[i]);
        free(nablay[i]);
    }
}



void evolve_wave_half(double *phi_in[NX], double *phi_out[NX], short int *xy_in[NX])
/* time step of field evolution */
{
    int i, j, iplus, iminus, jplus, jminus;
    double delta1, delta2, x, y;
    
    #pragma omp parallel for private(i,j,iplus,iminus,jplus,jminus,delta1,delta2,x,y)
    for (i=0; i<NX; i++){
        for (j=0; j<NY; j++){
            if (xy_in[i][j] == 1){
                /* discretized Laplacian depending on boundary conditions */
                if ((B_COND == BC_DIRICHLET)||(B_COND == BC_ABSORBING))
                {
                    iplus = (i+1);   if (iplus == NX) iplus = NX-1;
                    iminus = (i-1);  if (iminus == -1) iminus = 0;
                    jplus = (j+1);   if (jplus == NY) jplus = NY-1;
                    jminus = (j-1);  if (jminus == -1) jminus = 0;
                }
                else if (B_COND == BC_PERIODIC)
                {
                    iplus = (i+1) % NX;
                    iminus = (i-1) % NX;
                    if (iminus < 0) iminus += NX;
                    jplus = (j+1) % NY;
                    jminus = (j-1) % NY;
                    if (jminus < 0) jminus += NY;
                }
                
                delta1 = phi_in[iplus][j] + phi_in[iminus][j] + phi_in[i][jplus] + phi_in[i][jminus] - 4.0*phi_in[i][j];

                x = phi_in[i][j];

                /* evolve phi */
                if (B_COND != BC_ABSORBING)
                {
                    phi_out[i][j] = x + intstep*(delta1 - SPEED*(phi_in[iplus][j] - phi_in[i][j]));
                }
                else        /* case of absorbing b.c. - this is only an approximation of correct way of implementing */
                {
                    /* in the bulk */
                    if ((i>0)&&(i<NX-1)&&(j>0)&&(j<NY-1))
                    {
                        phi_out[i][j] = x - intstep*delta2;
                    }
                     /* right border */
                    else if (i==NX-1) 
                    {
                        phi_out[i][j] = x - intstep1*(x - phi_in[i-1][j]);
                    }
                    /* upper border */
                    else if (j==NY-1) 
                    {
                        phi_out[i][j] = x - intstep1*(x - phi_in[i][j-1]);
                    }
                    /* left border */
                    else if (i==0) 
                    {
                        phi_out[i][j] = x - intstep1*(x - phi_in[1][j]);
                    }
                   /* lower border */
                    else if (j==0) 
                    {
                        phi_out[i][j] = x - intstep1*(x - phi_in[i][1]);
                    }
                }


                if (FLOOR)
                {
                    if (phi_out[i][j] > VMAX) phi_out[i][j] = VMAX;
                    if (phi_out[i][j] < -VMAX) phi_out[i][j] = -VMAX;
                }
            }
        }
    }
    
//     printf("phi(0,0) = %.3lg, psi(0,0) = %.3lg\n", phi[NX/2][NY/2], psi[NX/2][NY/2]);
}

void evolve_wave(double *phi[NX], double *phi_tmp[NX], short int *xy_in[NX])
/* time step of field evolution */
{
    evolve_wave_half(phi, phi_tmp, xy_in);
    evolve_wave_half(phi_tmp, phi, xy_in);
}




double compute_variance(double *phi[NX], short int * xy_in[NX])
/* compute the variance (total probability) of the field */
{
    int i, j, n = 0;
    double variance = 0.0;

    for (i=1; i<NX; i++)
        for (j=1; j<NY; j++)
        {
            if (xy_in[i][j])
            {
                n++;
                variance += phi[i][j]*phi[i][j];
            }
        }
    if (n==0) n=1;
    return(variance/(double)n);
}

void renormalise_field(double *phi[NX], short int * xy_in[NX], double variance)
/* renormalise variance of field */
{
    int i, j;
    double stdv;
    
    stdv = sqrt(variance);

    for (i=1; i<NX; i++)
        for (j=1; j<NY; j++)
        {
            if (xy_in[i][j])
            {
                phi[i][j] = phi[i][j]/stdv;
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


void print_Julia_parameters()
{
    double pos[2];
    char message[50];
    
    glColor3f(1.0, 1.0, 1.0);
    if (julia_y >= 0.0) sprintf(message, "c = %.5f + %.5f i", julia_x, julia_y);
    else sprintf(message, "c = %.5f %.5f i", julia_x, julia_y);
    xy_to_pos(XMIN + 0.1, YMAX - 0.2, pos);
    write_text(pos[0], pos[1], message);
}

void set_Julia_parameters(int time, double *phi[NX], short int *xy_in[NX])
{
    double jangle, cosj, sinj, radius = 0.15;

    jangle = (double)time*DPI/(double)NSTEPS;
//     jangle = (double)time*0.001;
//     jangle = (double)time*0.0001;

    cosj = cos(jangle);
    sinj = sin(jangle);
    julia_x = -0.9 + radius*cosj;
    julia_y = radius*sinj;
    init_julia_set(phi, xy_in);
    
    printf("Julia set parameters : i = %i, angle = %.5lg, cx = %.5lg, cy = %.5lg \n", time, jangle, julia_x, julia_y);
}

void set_Julia_parameters_cardioid(int time, double *phi[NX], short int *xy_in[NX])
{
    double jangle, cosj, sinj, yshift;

    jangle = pow(1.05 + (double)time*0.00003, 0.333);
    yshift = 0.02*sin((double)time*PID*0.002);
//     jangle = pow(1.0 + (double)time*0.00003, 0.333);
//     jangle = pow(0.05 + (double)time*0.00003, 0.333);
//     jangle = pow(0.1 + (double)time*0.00001, 0.333);
//     yshift = 0.04*sin((double)time*PID*0.002);

    cosj = cos(jangle);
    sinj = sin(jangle);
    julia_x = 0.5*(cosj*(1.0 - 0.5*cosj) + 0.5*sinj*sinj);
    julia_y = 0.5*sinj*(1.0-cosj) + yshift;
//     julia_x = 0.5*(cosj*(1.0 - 0.5*cosj) + 0.5*sinj*sinj);
//     julia_y = 0.5*sinj*(1.0-cosj);
    init_julia_set(phi, xy_in);
    
    printf("Julia set parameters : i = %i, angle = %.5lg, cx = %.5lg, cy = %.5lg \n", time, jangle, julia_x, julia_y);
}

void animation()
{
    double time, scale, dx, var, jangle, cosj, sinj;
    double *phi[NX], *phi_tmp[NX];
    short int *xy_in[NX];
    int i, j, s;

    /* Since NX and NY are big, it seemed wiser to use some memory allocation here */
    for (i=0; i<NX; i++)
    {
        phi[i] = (double *)malloc(NY*sizeof(double));
        phi_tmp[i] = (double *)malloc(NY*sizeof(double));
        xy_in[i] = (short int *)malloc(NY*sizeof(short int));
    }

    npolyline = init_polyline(MDEPTH, polyline);
    for (i=0; i<npolyline; i++) printf("vertex %i: (%.3f, %.3f)\n", i, polyline[i].x, polyline[i].y);

    dx = (XMAX-XMIN)/((double)NX);
    intstep = DT/(dx*dx*VISCOSITY);
    intstep1 = DT/(dx*VISCOSITY);
    
//     julia_x = 0.1; 
//     julia_y = 0.6; 
    
//     set_Julia_parameters(0, phi, xy_in);
    
    printf("Integration step %.3lg\n", intstep);

    /* initialize wave wave function */
    init_gaussian(-1.0, 0.0, 0.1, 0.0, 0.01, phi, xy_in);
//     init_gaussian(x, y, mean, amplitude, scalex, phi, xy_in)
    
    if (SCALE)
    {
        var = compute_variance(phi, xy_in);
        scale = sqrt(1.0 + var);
        renormalise_field(phi, xy_in, var);
    }

    blank();
    glColor3f(0.0, 0.0, 0.0);
    

    glutSwapBuffers();
    
    draw_wave(phi, xy_in, 1.0, 0);
    if (DRAW_BILLIARD) draw_billiard(0, 1.0);
//     print_Julia_parameters(i);
    
//     print_level(MDEPTH);

    glutSwapBuffers();

    sleep(SLEEP1);
    if (MOVIE) for (i=0; i<SLEEP1*25; i++) save_frame();

    for (i=0; i<=NSTEPS; i++)
    {
        /* compute the variance of the field to adjust color scheme */
        /* the color depends on the field divided by sqrt(1 + variance) */
        if (SCALE)
        {
            var = compute_variance(phi, xy_in);
            scale = sqrt(1.0 + var);
//             printf("Norm: %5lg\t Scaling factor: %5lg\n", var, scale);
            renormalise_field(phi, xy_in, var);
        }
        else scale = 1.0;
        
        draw_wave(phi, xy_in, scale, i);
        
        for (j=0; j<NVID; j++) evolve_wave(phi, phi_tmp, xy_in);

        if (DRAW_BILLIARD) draw_billiard(0, 1.0);
        
//         print_level(MDEPTH);
//         print_Julia_parameters(i);

	glutSwapBuffers();
        
        /* modify Julia set */
//         set_Julia_parameters(i, phi, xy_in);

	if (MOVIE)
        {
            save_frame();

            /* it seems that saving too many files too fast can cause trouble with the file system */
            /* so this is to make a pause from time to time - parameter PAUSE may need adjusting   */
            if (i % PAUSE == PAUSE - 1)
            {
                printf("Making a short pause\n");
                sleep(PSLEEP);
                s = system("mv wave*.tif tif_heat/");
            }
        }

    }

    if (MOVIE)
    {
        for (i=0; i<20; i++) save_frame();
        s = system("mv wave*.tif tif_heat/");
    }
    for (i=0; i<NX; i++)
    {
        free(phi[i]);
        free(phi_tmp[i]);
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
    glutCreateWindow("Heat equation in a planar domain");

    init();

    glutDisplayFunc(display);

    glutMainLoop();

    return 0;
}

