/* Global variables and parameters for wave_billiard, heat and schrodinger */

// #include "hsluv.c"


/* Basic math */

#define PI 	3.141592654
#define DPI 	6.283185307
#define PID 	1.570796327

/* shape of domain */

#define D_NOTHING 999   /* no boundaries */
#define D_RECTANGLE 0   /* rectangular domain */
#define D_ELLIPSE 1     /* elliptical domain */
#define D_EXT_ELLIPSE 199   /* exterior of elliptical domain */
#define D_EXT_ELLIPSE_CURVED 198   /* exterior of curved elliptical domain */
#define D_EXT_ELLIPSE_CURVED_BDRY 197   /* exterior of curved elliptical domain, with horizontal boundaries */
#define D_STADIUM 2     /* stadium-shaped domain */
#define D_SINAI 3       /* Sinai billiard */
#define D_DIAMOND 4     /* diamond-shaped billiard */
#define D_TRIANGLE 5    /* triangular billiard */
#define D_FLAT 6        /* flat interface */
#define D_ANNULUS 7     /* annulus */
#define D_POLYGON 8     /* polygon */
#define D_YOUNG 9       /* Young diffraction slits */
#define D_GRATING 10    /* diffraction grating */
#define D_EHRENFEST 11  /* Ehrenfest urn type geometry */

/* The following 3 types are superseded by D_CIRCLES and can be removed later */
#define D_DISK_GRID 12  /* grid of disks */
#define D_DISK_HEX 13   /* haxagonl grid of disks */
#define D_DISK_PERCOL 14    /* random grid of percolation type */

#define D_MENGER 15     /* Menger-Sierpinski carpet */ 
#define D_JULIA_INT 16  /* interior of Julia set */ 
#define D_MENGER_ROTATED 17  /* rotated Menger-Sierpinski carpet */
#define D_PARABOLA 18   /* parabolic domain */
#define D_TWO_PARABOLAS 19   /* two facing parabolic antennas */

#define D_CIRCLES 20    /* several circles */
#define D_CIRCLES_IN_RECT 201   /* several circles in a rectangle */

#define D_FOUR_PARABOLAS 31     /* four parabolas with axes in NSEW directions */
#define D_POLY_PARABOLAS 32     /* polygon with parabolic sides */
#define D_PENROSE 33            /* Penrose unilluminable room */
#define D_HYPERBOLA 34          /* one branch of hyperbola */
#define D_TOKARSKY 35           /* Tokarsky unilluminable room */
#define D_TOKA_PRIME 36         /* Tokarsky room made of 86 triangles */
#define D_ISOSPECTRAL 37        /* isospectral billiards */
#define D_HOMOPHONIC 38         /* homophonic billiards */

#define D_POLYGONS 40           /* several polygons */
#define D_VONKOCH 41            /* von Koch snowflake fractal */
#define D_STAR 42               /* star shape */
#define D_FRESNEL 43            /* Fresnel lens */
#define D_NOISEPANEL 44         /* zigzag noise insulating panel */
#define D_NOISEPANEL_RECT 441   /* comparison between zigzag noise insulating panel and flat walls */
#define D_DOUBLE_FRESNEL 45     /* two facing Fresnel lenses */
#define D_QRD 46                /* quadratic resonance diffuser */
#define D_QRD_ASYM 461          /* asymmetric quadratic resonance diffuser */
#define D_CIRCLE_SEGMENT 47     /* lens-shaped circular segment */
#define D_GROOVE 48             /* groove array supposed to induce polaritons */
#define D_FABRY_PEROT 49        /* Fabry-Perrot cavity (in fact simply a vertical slab) */
#define D_LSHAPE 50             /* L-shaped billiard (surface of genus 2) */
#define D_WAVEGUIDE 51          /* wave guide */
#define D_WAVEGUIDE_W 52        /* W-shaped wave guide */
#define D_MAZE 53               /* maze */
#define D_MAZE_CLOSED 54        /* closed maze */
#define D_MAZE_CHANNELS 541     /* maze with two channels attached */
#define D_CHESSBOARD 55         /* chess board configuration */
#define D_TRIANGLE_TILES 56     /* triangular tiling */
#define D_HEX_TILES 57          /* honeycomb tiling */
#define D_FUNNELS 58            /* two funnels */
#define D_ONE_FUNNEL 581        /* one funnel */
#define D_LENSES_RING 59        /* several lenses forming a ring */
#define D_MAZE_CIRCULAR 60      /* circular maze */

#define D_WING 70               /* complement of wing-shaped domain */
#define D_TESLA 71              /* Tesla valve */

#define NMAXCIRCLES 10000       /* total number of circles/polygons (must be at least NCX*NCY for square grid) */
#define NMAXPOLY 50000          /* maximal number of vertices of polygonal lines (for von Koch et al) */

#define C_SQUARE 0          /* square grid of circles */
#define C_HEX 1             /* hexagonal/triangular grid of circles */
#define C_RAND_DISPLACED 2  /* randomly displaced square grid */
#define C_RAND_PERCOL 3     /* random percolation arrangement */
#define C_RAND_POISSON 4    /* random Poisson point process */
#define C_CLOAK 5           /* invisibility cloak */
#define C_CLOAK_A 6         /* first optimized invisibility cloak */
#define C_LASER 7           /* laser fight in a room of mirrors */
#define C_POISSON_DISC 8    /* Poisson disc sampling */

#define C_GOLDEN_MEAN 10    /* pattern based on vertical shifts by golden mean */
#define C_GOLDEN_SPIRAL 11  /* spiral pattern based on golden mean */
#define C_SQUARE_HEX 12     /* alternating between square and hexagonal/triangular */
#define C_HEX_NONUNIF 13    /* triangular grid with non-constant column distance */

#define C_RINGS 20          /* obstacles arranged in concentric rings */
#define C_RINGS_T 201       /* obstacles arranged in concentric rings, triangular lattice */
#define C_RINGS_SPIRAL 202  /* obstacles arranged on a "subflower" spiral, similar to C_GOLDEN_SPIRAL */

#define C_ONE 97            /* one single circle, as for Sinai */
#define C_TWO 98            /* two concentric circles of different type */
#define C_NOTHING 99        /* no circle at all, for comparisons */


/* Billiard tables for heat equation */

#define D_ANNULUS_HEATED 21 /* annulus with different temperatures */
#define D_MENGER_HEATED 22  /* Menger gasket with different temperatures */
#define D_MENGER_H_OPEN 23  /* Menger gasket with different temperatures and larger domain */
#define D_MANDELBROT 24     /* Mandelbrot set */
#define D_JULIA 25          /* Julia set */
#define D_MANDELBROT_CIRCLE 26     /* Mandelbrot set with circular conductor */
#define D_VONKOCH_HEATED 27 /* von Koch snowflake in larger circle */

/* Variable index of refraction */
#define IOR_MANDELBROT 1      /* index of refraction depends on escape time in Mandelbrot set (log) */
#define IOR_MANDELBROT_LIN 100    /* index of refraction depends on escape time in Mandelbrot set (linear) */
#define IOR_EARTH 2         /* index of refraction models speed of seismic waves */
#define IOR_EXPLO_LENSING 3 /* explosive lensing */
#define IOR_PERIODIC_WELLS 4  /* periodic superposition of "wells" */
#define IOR_RANDOM_WELLS 5  /* random superposition of "wells" */
#define IOR_PERIODIC_WELLS_ROTATING 6   /* periodic superposition rotating in time */
#define IOR_PERIODIC_WELLS_ROTATING_LARGE 7   /* periodic superposition rotating in time, larger area */

/* Boundary conditions */

#define BC_DIRICHLET 0   /* Dirichlet boundary conditions */
#define BC_PERIODIC 1    /* periodic boundary conditions */
#define BC_ABSORBING 2   /* absorbing boundary conditions (beta version) */
#define BC_VPER_HABS 3   /* vertically periodic and horizontally absorbing boundary conditions */
#define BC_ABS_REFLECT 4   /* absorbing boundary conditions, except reflecting at y=0, for comparisons */
#define BC_LSHAPE 10      /* L-shaped boundary conditions (surface of genus 2) */
// #define BC_OSCILL_ABSORB 5  /* oscillating boundary condition on the left, absorbing on other walls */ 


/* Oscillating boundary conditions */

#define OSC_PERIODIC 0  /* periodic oscillation */
#define OSC_SLOWING 1   /* oscillation of slowing frequency (anti-chirp) */
#define OSC_WAVE_PACKET 2   /* Gaussian wave packet */
#define OSC_CHIRP 3     /* chirp (linearly accelerating frequency) */

/* For debugging purposes only */
// #define FLOOR 0         /* set to 1 to limit wave amplitude to VMAX */
// #define VMAX 10.0       /* max value of wave amplitude */

/* Plot types */

/* For wave equation */
#define P_AMPLITUDE 0    /* plot amplitude of wave */
#define P_ENERGY 1       /* plot energy of wave */
#define P_MIXED 2        /* plot amplitude in upper half, energy in lower half */
#define P_MEAN_ENERGY 3  /* energy averaged over time */
#define P_LOG_ENERGY 4   /* log of energy averaged over time */
#define P_LOG_MEAN_ENERGY 5  /* log of energy averaged over time */
#define P_ENERGY_FLUX 6     /* energy flux */
#define P_TOTAL_ENERGY_FLUX 7    /* energy flux averaged over time */

/* For Schrodinger equation */
#define P_MODULE 10        /* plot module of wave function squared */
#define P_PHASE 11         /* plot phase of wave function */
#define P_REAL 12          /* plot real part */
#define P_IMAGINARY 13     /* plot imaginary part */


/* Color schemes */

#define C_LUM 0          /* color scheme modifies luminosity (with slow drift of hue) */
#define C_HUE 1          /* color scheme modifies hue */
#define C_PHASE 2        /* color scheme shows phase */
#define C_ONEDIM 3       /* use preset 1d color scheme (for Turbo, Viridis, Magma, Inferno, Plasma) */
#define C_ONEDIM_LINEAR 4   /* use preset 1d color scheme with linear scale */

/* Color palettes */

#define COL_JET 0       /* JET color palette */
#define COL_HSLUV 1     /* HSLUV color palette (perceptually uniform) */
#define COL_GRAY 2      /* grayscale */

#define COL_TURBO 10     /* TURBO color palette (by Anton Mikhailov) */
#define COL_VIRIDIS 11   /* Viridis color palette */
#define COL_MAGMA 12     /* Magma color palette */
#define COL_INFERNO 13   /* Inferno color palette */
#define COL_PLASMA 14    /* Plasma color palette */
#define COL_CIVIDIS 15   /* Cividis color palette */
#define COL_PARULA 16    /* Parula color palette */
#define COL_TWILIGHT 17  /* Twilight color palette */
#define COL_TWILIGHT_SHIFTED 18  /* Shifted twilight color palette */

#define COL_TURBO_CYCLIC 101    /* TURBO color palette (by Anton Mikhailov) corrected to be cyclic, beta */

#define NPWIDTH 0.02      /* width of noise panel separation */


/* plot types used by rde */

#define Z_AMPLITUDE 0   /* amplitude of first field */
#define Z_RGB 20        /* RGB plot */
#define Z_POLAR 21      /* polar angle associated with RBG plot */
#define Z_NORM_GRADIENT 22      /* gradient of polar angle */
#define Z_ANGLE_GRADIENT 221    /* direction of polar angle */
#define Z_NORM_GRADIENTX 23     /* norm of gradient of u */
#define Z_ANGLE_GRADIENTX 231   /* direction of gradient of u */
#define Z_NORM_GRADIENT_INTENSITY 24  /* gradient and intensity of polar angle */
#define Z_VORTICITY 25  /* curl of polar angle */
#define Z_VORTICITY_ABS 251  /* absolute value of curl of polar angle */

/* for Schrodinger equation */
#define Z_MODULE 30       /* module squared of first two fields */
#define Z_ARGUMENT 31     /* argument of first two fields, with luminosity depending on module */
#define Z_REALPART 32     /* first field, with luminosity depending on module */

/* for RPSLZ equation */
#define Z_MAXTYPE_RPSLZ 40      /* color of type with maximal density */
#define Z_THETA_RPSLZ 41        /* polar angle */
#define Z_NORM_GRADIENT_RPSLZ 42    /* gradient of polar angle */

/* for Euler incompressible Euler equation */
#define Z_EULER_VORTICITY 50    /* vorticity of velocity */
#define Z_EULER_LOG_VORTICITY 51    /* log of vorticity of velocity */
#define Z_EULER_VORTICITY_ASYM 52   /* vorticity of velocity */
#define Z_EULER_LPRESSURE 53        /* Laplacian of pressure */
#define Z_EULER_PRESSURE 54         /* pressure */

/* for Euler compressible Euler equation */
#define Z_EULER_DENSITY 60           /* density */
#define Z_EULER_SPEED 61             /* norm of velocity */
#define Z_EULERC_VORTICITY 62        /* vorticity of velocity */
#define Z_EULER_DIRECTION 63         /* direction of velocity */
#define Z_EULER_DIRECTION_SPEED 64   /* hut for direction of velocity, luminosity for speed */

/* special boundary conditions for Euler equation */
#define BCE_TOPBOTTOM 1         /* laminar flow at top and bottom */
#define BCE_TOPBOTTOMLEFT 2     /* laminar flow at top, bottom and left side */
#define BCE_CHANNELS 3          /* laminar flow in channels at left and right */
#define BCE_MIDDLE_STRIP 4      /* laminar flow in horizontal strip in the middle */
#define BCE_LEFT 5              /* laminar flow at left side */

typedef struct
{
    double xc, yc, radius;      /* center and radius of circle */
    short int active, top;      /* circle is active, circle is in top half */
} t_circle;

typedef struct
{
    double xc, yc, radius, angle;      /* center, radius and angle of polygon */
    int nsides;                         /* number of sides of polygon */
    short int active, top;              /* polygon is active, polygon is in top half */
} t_polygon;

typedef struct
{
    double x, y;           /* (x,y) coordinates of vertex */
    double posi, posj;     /* (i,j) coordinates of vertex */
} t_vertex;

typedef struct
{
    double x1, y1, x2, y2;   /* (x,y) coordinates of vertices */
    double posi1, posj1, posi2, posj2;   /* (i,j) coordinates of vertices */
} t_rectangle;

typedef struct
{
    double x1, y1, x2, y2;   /* (x,y) coordinates of long symmetry axis */
    double width;            /* width of rectangle */
    double posi1, posj1, posi2, posj2;   /* (i,j) coordinates of vertices */
    double posi3, posj3, posi4, posj4;   /* (i,j) coordinates of vertices */
} t_rect_rotated;

typedef struct
{
    double xc, yc;           /* (x,y) coordinates of center */
    double r, width;         /* radius and width of arc */
    double angle1, dangle;   /* start angle and angular width */
    double posi1, posj1;     /* (i,j) coordinates of center */
//     double posi3, posj3, posi4, posj4;   /* (i,j) coordinates of vertices */
} t_arc;

typedef struct
{
    int nneighb;    /* number of neighbours to compute Laplacian */
    double *nghb[4];    /* pointers to neighbours */ 
} t_laplacian;


typedef struct
{
    double xc, yc;           /* (x,y) coordinates of center */
    int ix, iy;              /* lattice coordinates of center */
    double period, amp;      /* period and amplitude */
    double phase;            /* phase shift */
    double var_envelope;     /* variance of Gaussian envelope */
    int time_shift;          /* time shift */
} t_wave_packet;



int ncircles = NMAXCIRCLES;         /* actual number of circles, can be decreased e.g. for random patterns */
int npolyline = NMAXPOLY;           /* actual length of polyline */
int npolyrect = NMAXPOLY;           /* actual number of polyrect */
int npolyrect_rot = NMAXPOLY;       /* actual number of rotated polyrect */
int npolyarc = NMAXPOLY;            /* actual number of arcs */

t_circle circles[NMAXCIRCLES];      /* circular scatterers */
t_polygon polygons[NMAXCIRCLES];    /* polygonal scatterers */
t_vertex polyline[NMAXPOLY];        /* vertices of polygonal line */
t_rectangle polyrect[NMAXPOLY];     /* vertices of rectangles */
t_rect_rotated polyrectrot[NMAXPOLY];  /* data of rotated rectangles */
t_arc polyarc[NMAXPOLY];            /* data of arcs */

/* the same for comparisons between different domains */
int ncircles_b = NMAXCIRCLES;         /* actual number of circles, can be decreased e.g. for random patterns */
int npolyline_b = NMAXPOLY;           /* actual length of polyline */
t_circle circles_b[NMAXCIRCLES];      /* circular scatterers */
t_polygon polygons_b[NMAXCIRCLES];    /* polygonal scatterers */
t_vertex polyline_b[NMAXPOLY];        /* vertices of polygonal line */

// double julia_x = -0.5, julia_y = 0.5;    /* parameters for Julia sets */
// double julia_x = 0.33267, julia_y = 0.06395;    /* parameters for Julia sets */
double julia_x = 0.37468, julia_y = 0.21115;    /* parameters for Julia sets */
