/* Global variables and parameters for lennardjones */

/* Basic math */

#define PI 	3.141592654
#define DPI 	6.283185307
#define PID 	1.570796327

/* shape of domain */

#define D_CIRCLES 20    /* several circles */
#define D_CIRCLES_IN_RECT 201   /* several circles in a rectangle */

#define NMAXCIRCLES 100000       /* total number of circles/polygons (must be at least NCX*NCY for square grid) */
#define MAXNEIGH 20         /* max number of neighbours kept in memory */
#define NMAXOBSTACLES 100   /* max number of obstacles */
#define NMAXSEGMENTS 1000   /* max number of repelling segments */
#define NMAXGROUPS 50       /* max number of groups of segments */
#define NMAXCOLLISIONS 200000   /* max number of collisions */

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

#define C_POOL_TABLE 20     /* pool table initial position */

#define C_ONE 97            /* one single circle, as for Sinai */
#define C_TWO 98            /* two concentric circles of different type */
#define C_NOTHING 99        /* no circle at all, for comparisons */

/* pattern of additional obstacles */
#define O_CORNERS 0         /* obstacles in the corners (for Boy b.c.) */
#define O_GALTON_BOARD 1    /* Galton board pattern */
#define O_GENUS_TWO 2       /* obstacles in corners of L-shape domeain (for genus 2 b.c.) */
#define O_POOL_TABLE 3      /* obstacles around pockets of pool table */
#define O_HLINE_HOLE_SPOKES 181    /* tips of spokes for S_HLINE_HOLE_SPOKES segment pattern */

/* pattern of additional repelling segments */
#define S_RECTANGLE 0       /* segments forming a rectangle */
#define S_CUP 1             /* segments forming a cup (for increasing gravity) */
#define S_HOURGLASS 2       /* segments forming an hour glass */
#define S_PENTA 3           /* segments forming a pentagon with 3 angles of 120° and 2 right angles */
#define S_CENTRIFUGE 4      /* segments forming "centrifuge" (polygon with radial segments) */
#define S_POLY_ELLIPSE 5    /* segments forming a polygonal approximation of an ellipse */
#define S_POOL_TABLE 6      /* pool table with pockets */
#define S_CENTRIFUGE_RND 7  /* segments forming centrifuge with more rounded bins */
#define S_CENTRIFUGE_LEAKY 8  /* segments forming centrifuge with rounded bins and holes */
#define S_CIRCLE_EXT 9      /* segments forming a repelling cicle */
#define S_ROCKET_NOZZLE 10  /* segments forming a rocket with bell-shaped nozzle */
#define S_ROCKET_NOZZLE_ROTATED 101 /* rotated version of rocket with bell-shaped nozzle */
#define S_TWO_ROCKETS 102     /* two different rockets, with nozzles specified by NOZZLE_SHAPE and NOZZLE_SHAPE_B */
#define S_TWO_CIRCLES_EXT 11  /* segments forming two repelling cicle */
#define S_DAM 12              /* segments forming a dam that can break */
#define S_DAM_WITH_HOLE 13    /* segments forming a dam in which a hole can open */
#define S_DAM_WITH_HOLE_AND_RAMP 14    /* segments forming a dam in which a hole can open */
#define S_MAZE 15           /* segments forming a maze */
#define S_EXT_RECTANGLE 16  /* particles outside a rectangle */
#define S_DAM_BRICKS 17     /* dam made of several bricks */
#define S_HLINE_HOLE 18    /* horizontal line with a hole in the bottom */
#define S_HLINE_HOLE_SPOKES 181    /* horizontal line with a hole in the bottom and extra spokes */
#define S_EXT_CIRCLE_RECT 19    /* particles outside a circle and a rectangle */
#define S_BIN_OPENING 20        /* bin containing particles opening at deactivation time */

/* particle interaction */

#define I_COULOMB 0         /* Coulomb force */
#define I_LENNARD_JONES 1   /* Lennard-Jones force */
#define I_LJ_DIRECTIONAL 2  /* Lennard-Jones with direction dependence of square symmetry */
#define I_LJ_PENTA 3        /* Lennard-Jones with pentagonal symmetry */
#define I_GOLDENRATIO 4     /* Lennard-Jones type with equilibria related by golden ratio */
#define I_LJ_DIPOLE 5       /* Lennard-Jones with a dipolar angle dependence */
#define I_LJ_QUADRUPOLE 6   /* Lennard-Jones with a quadropolar angle dependence */
#define I_LJ_WATER 7        /* model for water molecule */

/* Boundary conditions */

#define BC_SCREEN 0         /* harmonic boundary conditions outside screen area */
#define BC_RECTANGLE 1      /* harmonic boundary conditions on a resizeable rectangle */
#define BC_CIRCLE 2         /* harmonic boundary conditions outside a moving circle */
#define BC_PERIODIC 3       /* periodic boundary conditions */
#define BC_PERIODIC_CIRCLE 4  /* periodic boundary conditions and harmonic b.c. outside moving circle */
#define BC_EHRENFEST 5      /* Ehrenfest urn-type configuration */
#define BC_PERIODIC_FUNNEL 6    /* funnel with periodic boundary conditions */
#define BC_RECTANGLE_LID 7  /* rectangular container with moving lid */
#define BC_PERIODIC_TRIANGLE 8  /* periodic boundary conditions and harmonic b.c. outside moving triangle */
#define BC_RECTANGLE_WALL 9 /* rectangular container with vertical movable wall */
#define BC_KLEIN 11         /* Klein bottle (periodic with twisted vertical parts) */
#define BC_SCREEN_BINS 12   /* harmonic boundary conditions outside screen area plus "bins" (for Galton board) */
#define BC_BOY 13           /* Boy surface/projective plane (periodic with twisted horizontal and vertical parts) */
#define BC_GENUS_TWO 14     /* surface of genus 2, obtained by identifying opposite sides of an L shape */
#define BC_ABSORBING 20     /* "no-return" boundary conditions outside BC area */
#define BC_REFLECT_ABS 21   /* reflecting on lower boundary, and "no-return" boundary conditions outside BC area */

/* Regions for partial thermostat couplings */

#define TH_VERTICAL 0       /* only particles at the right of x = PARTIAL_THERMO_SHIFT are coupled */
#define TH_INSEGMENT 1      /* only particles in region defined by segments are coupled */
#define TH_INBOX 2          /* only particles in a given box are coupled */
#define TH_LAYER 3          /* only particles above -LAMBDA are coupled */
#define TH_LAYER_TYPE2 4    /* only particles above highest type 2 particle are coupled */

/* Gravity schedules */

#define G_INCREASE_RELEASE 1    /* slow increase and instant release */
#define G_INCREASE_DECREASE 2   /* slow increase an decrease */

/* Rocket shapes */

#define RCK_DISC 0      /* disc-shaped rocket */
#define RCK_RECT 1      /* rectangular rocket */
#define RCK_RECT_HAT 2  /* rectangular rocket with a hat */

/* Nozzle shapes */

#define NZ_STRAIGHT 0   /* straight nozzle */
#define NZ_BELL 1       /* bell-shaped nozzle */
#define NZ_GLAS 2       /* glas-shaped nozzle */
#define NZ_CONE 3       /* cone-shaped nozzle */
#define NZ_TRUMPET 4    /* trumpet-shaped nozzle */
#define NZ_BROAD 5      /* broad straight nozzle */
#define NZ_NONE 99      /* no nozzle */

/* Types of chemical reactions */

#define CHEM_RPS 0      /* rock-paper-scissors reaction */
#define CHEM_AAB 1      /* reaction A + A -> B */
#define CHEM_ABC 2      /* reaction A + B -> C */
#define CHEM_A2BC 3      /* reaction 2A + B -> C */
#define CHEM_CATALYSIS 4    /* reaction 2A + C -> B + C */
#define CHEM_BAA 5      /* reaction B -> A + A (dissociation) */
#define CHEM_AABAA 6    /* reaction A + A <-> B (reversible) */
#define CHEM_POLYMER 7  /* reaction A + B -> C, A + C -> D, etc */
#define CHEM_POLYMER_DISS 8  /* polimerisation with dissociation */
#define CHEM_POLYMER_STEP 9  /* step growth polimerisation with dissociation */
#define CHEM_AUTOCATALYSIS 10   /* autocatalytic reaction 2A + B -> 2B */
#define CHEM_CATALYTIC_A2D 11   /* catalytic reaction A + B -> C, A + C -> B + D */
#define CHEM_ABCAB 12       /* reaction A + B <-> C (reversible) */
#define CHEM_ABCDABC 13     /* reactions A + B <-> C, A + C <-> D */
#define CHEM_BZ 14          /* simplified Belousov-Zhabotinski reaction with 6 types (Oregonator) */
#define CHEM_BRUSSELATOR 15 /* Brusselator oscillating reaction */

/* Initial conditions for chemical reactions */

#define IC_UNIFORM 0       /* all particles have type 1 */
#define IC_UNIFORM2 20     /* all particles have type 2 */
#define IC_RANDOM_UNIF 1   /* particle type chosen uniformly at random */
#define IC_RANDOM_TWO 2    /* particle type chosen randomly between 1 and 2, with TYPE_PROPORTION */
#define IC_CIRCLE 3        /* type 1 in a disc */
#define IC_CATALYSIS 4     /* mix of 1 and 2 in left half, only 1 in right half */
#define IC_LAYERS 5        /* layer of 2 below 1 */
#define IC_BZ 6            /* initial state for BZ reaction */

/* Plot types */

#define P_KINETIC 0       /* colors represent kinetic energy of particles */
#define P_NEIGHBOURS 1    /* colors represent number of neighbours */
#define P_HEALTH 2        /* colors represent health (for SIR model) */
#define P_BONDS 3         /* draw lattice based on neighbours */
#define P_ANGLE 4         /* colors represent angle/spin of particle */
#define P_TYPE 5          /* colors represent type of particle */
#define P_DIRECTION 6     /* colors represent direction of velocity */
#define P_ANGULAR_SPEED 7 /* colors represent angular speed */
#define P_DIRECT_ENERGY 8 /* hues represent direction, luminosity represents energy */
#define P_DIFF_NEIGHB 9   /* colors represent number of neighbours of different type */
#define P_THERMOSTAT 10   /* colors show which particles are coupled to the thermostat */
#define P_INITIAL_POS 11  /* colors depend on initial position of particle */

/* Color schemes */

#define C_LUM 0          /* color scheme modifies luminosity (with slow drift of hue) */
#define C_HUE 1          /* color scheme modifies hue */
#define C_PHASE 2        /* color scheme shows phase */
#define C_ONEDIM 3       /* use preset 1d color scheme (for Turbo, Viridis, Magma, Inferno, Plasma, Twilight) */
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


typedef struct
{
    double xc, yc, radius;      /* center and radius of circle */
    double angle;               /* angle of particle's "spin" */
    short int active;           /* circle is active */
    double energy;              /* dissipated energy */
    double vx;                  /* x velocity of particle */
    double vy;                  /* y velocity of particle */
    double omega;               /* angular velocity of particle */
    double mass_inv;            /* inverse of particle mass */
    double inertia_moment_inv;  /* inverse of moment of inertia */
    double fx;                  /* x component of force on particle */
    double fy;                  /* y component of force on particle */
    double torque;              /* torque on particle */
    short int thermostat;       /* whether particle is coupled to thermostat */
    int hashcell;               /* hash cell in which particle is located */
    int neighb;                 /* number of neighbours within given distance */
    int diff_neighb;            /* number of neighbours of different type */
    int hash_nneighb;           /* number of neighbours in hashgrid */
    int hashneighbour[9*HASHMAX];   /* particle numbers of neighbours in hashgrid */
    double deltax[9*HASHMAX];   /* relative position of neighbours */
    double deltay[9*HASHMAX];   /* relative position of neighbours */
    short int type;             /* type of particle, for mixture simulations */
    short int interaction;      /* type of interaction */
    double eq_dist;             /* equilibrium distance */
    double spin_range;          /* range of spin-spin interaction */
    double spin_freq;           /* angular frequency of spin-spin interaction */
    double color_hue;           /* color hue of particle, for P_INITIAL_POS plot type */
    int color_rgb[3];           /* RGB colors code of particle, for use in ljones_movie.c */
} t_particle;

typedef struct
{
    int number;                 /* total number of particles in cell */
    int particles[HASHMAX];     /* numbers of particles in cell */
    int nneighb;                /* number of neighbouring cells */
    int neighbour[9];           /* numbers of neighbouring cells */
} t_hashgrid;

typedef struct
{
    double xc, yc, radius;      /* center and radius of circle */
    short int active;           /* circle is active */
    double energy;              /* dissipated energy */
    double vx;                  /* x velocity of particle */
    double vy;                  /* y velocity of particle */
    double mass_inv;            /* inverse of particle mass */
    double fx;                  /* x component of force on particle */
    double fy;                  /* y component of force on particle */
    int hashx;                  /* hash grid positions of particles */
    int hashy;                  /* hash grid positions of particles */
    int neighb;                 /* number of neighbours */
    int health;                 /* 0 = healthy, 1 = infected, 2 = recovered */
    double infected_time;       /* time since infected */
    int protected;              /* 0 = not protected, 1 = protected */
} t_person;

typedef struct
{
    double xc, yc, radius;      /* center and radius of circle */
    short int active;           /* circle is active */
} t_obstacle;

typedef struct
{
    double x1, x2, y1, y2;      /* extremities of segment */
    double xc, yc;              /* mid-point of segment */
    double nx, ny;              /* normal vector */
    double c;                   /* constant term in cartesian eq nx*x + ny*y = c */
    double length;              /* length of segment */
    short int concave;          /* corner is concave, to add extra repelling force */
    short int cycle;            /* set to 1 if (x2, y2) is equal to (x1, y1) of next segment */
    short int group;            /* group to which segment belongs (for several obstacles) */
    double angle1, angle2;      /* angles in which concave corners repel */
    short int active;           /* segment is active */
    double x01, x02, y01, y02;  /* initial values of extremities, in case of rotation/translation */
    double nx0, ny0;            /* initial normal vector */
    double angle01, angle02;    /* initial values of angles in which concave corners repel */
    double fx, fy;              /* x and y-components of force on segment */
    double torque;              /* torque on segment with respect to its center */
    short int inactivate;       /* set to 1 for segment to become inactive at time SEGMENT_DEACTIVATION_TIME */
} t_segment;

typedef struct
{
    double xc, yc;              /* center of circle */
} t_tracer;

typedef struct
{
    double xc, yc;              /* center of mass of obstacle */
    double angle;               /* orientation of obstacle */
    double vx, vy;              /* velocity of center of mass */
    double omega;               /* angular velocity */ 
    double mass;                /* mass of obstacle */
    double moment_inertia;      /* moment of inertia */
} t_group_segments;

typedef struct
{
    double xc, yc;              /* coordinates of centers of mass */
    double vx, vy;              /* velocities */
    double omega;               /* angular velocity */
} t_group_data;


typedef struct
{
    double x, y;                /* location of collision */
    int time;                   /* time since collision */
    int color;                  /* color hue in case of different collisions */
} t_collision;

int ncircles, nobstacles, nsegments, ngroups = 1, counter = 0;

