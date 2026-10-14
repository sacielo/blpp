/*
 * sod2d.c - a 2D compressible-Euler mini-app built on the blpp kernels.
 *
 * Sod shock tube on a uniform grid: dimensionally split first-order
 * Godunov with Rusanov (local Lax-Friedrichs) face fluxes, specular
 * reflection y-walls, zero-gradient x-boundaries, optional lap5 bulk
 * diffusion.  Conserved state U = (d, mx, my, E); the blpp 3-vector
 * fields are SoA component arrays, so momenta carry a z component that
 * stays 0 (2D flow = z-degenerate 3-vector field).  Domain [0,1]^2,
 * diaphragm at x = 0.5: (d,u,p) = (1,0,1) | (0.125,0,0.1).
 *
 * This is a dogfood consumer of v3blas.h / cblas.h: kernels where they
 * fit, standard BLAS where it already fits, plain loops for gathers,
 * reductions and per-face physics.  Kernel map:
 *   velocity v = m*(1/d)     v3had     (one scalar field supplied to
 *                                        all three component slots)
 *   kinetic energy           v3mom_ke  (ke = 0.5*d*(v.v) in one pass)
 *   |v|^2 (Mach report)      v3sqr
 *   scalar field products    v1xypa    (r = q.*t, a = 0: mass flux,
 *                                        energy flux, dissipation)
 *   wall reflection          v3refl    (q = m - 2(m.n)n with a true
 *                                        scalar-broadcast unit normal,
 *                                        eps = 0)
 *   bulk diffusion           cblas_dlap5 with jx=1, jy=nxg: the
 *                            stencil strides ARE the grid strides
 *   sums and scaling         cblas_daxpy / cblas_dscal
 * Global reductions (dt bound, mass, Mach) are plain loops on purpose:
 * no kernel here does a reduction, by design.
 *
 * usage: sod2d [-nx N] [-ny N] [-tend T] [-cfl C] [-av A] [-out K]
 * Writes sod2d.vtk (ASCII) at the end; progress every -out steps.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cblas.h"
#include "v3blas.h"

#define GAMMA 1.4
#define SMALL 1e-12

typedef struct {
    int nx, ny;                        /* interior cells                */
    int nxg, nyg, n;                   /* +1 ghost cell per side        */
    double dx, dy, dt;
    double *d, *mx, *my, *mz, *E;      /* conserved (flattened i+j*nxg) */
    double *invd, *p, *ke, *vsq;       /* primitives / derived          */
    double *vx, *vy, *vz;              /* velocity field                */
    double *km, *lap;                  /* scratch (one grid)            */
    int nfm;                           /* max faces per line            */
    double *L[4], *R[4];               /* face buffers: left/right U    */
    double *vnL, *vnR, *vtL, *vtR;     /* normal/transverse velocities  */
    double *pL, *pR;
    double *F[4], *du, *s, *t1, *t2;   /* face flux + work              */
} grid;

static void *xcalloc(size_t k)
{
    void *q = calloc(k, sizeof(double));
    if (!q) { fprintf(stderr, "sod2d: out of memory\n"); exit(3); }
    return q;
}

static void alloc_grid(grid *g, int nx, int ny)
{
    int k;
    g->nx = nx; g->ny = ny;
    g->nxg = nx + 2; g->nyg = ny + 2;
    g->n = g->nxg * g->nyg;
    g->dx = 1.0 / nx; g->dy = 1.0 / ny;
    g->d = xcalloc(g->n);  g->mx = xcalloc(g->n); g->my = xcalloc(g->n);
    g->mz = xcalloc(g->n); g->E = xcalloc(g->n);
    g->invd = xcalloc(g->n); g->p = xcalloc(g->n);
    g->ke = xcalloc(g->n); g->vsq = xcalloc(g->n);
    g->vx = xcalloc(g->n); g->vy = xcalloc(g->n); g->vz = xcalloc(g->n);
    g->km = xcalloc(3 * (size_t)g->n); g->lap = xcalloc(g->n);
    g->nfm = (g->nxg > g->nyg ? g->nxg : g->nyg) + 1;
    for (k = 0; k < 4; k++) { g->L[k] = xcalloc(g->nfm);
                              g->R[k] = xcalloc(g->nfm);
                              g->F[k] = xcalloc(g->nfm); }
    g->vnL = xcalloc(g->nfm); g->vnR = xcalloc(g->nfm);
    g->vtL = xcalloc(g->nfm); g->vtR = xcalloc(g->nfm);
    g->pL = xcalloc(g->nfm);  g->pR = xcalloc(g->nfm);
    g->du = xcalloc(g->nfm);  g->s = xcalloc(g->nfm);
    g->t1 = xcalloc(g->nfm);  g->t2 = xcalloc(g->nfm);
}

#define AT(arr, j) ((arr) + (size_t)(j) * (size_t)g->nxg)
#define UC(k) ((k) == 0 ? g->d : (k) == 1 ? g->mx : (k) == 2 ? g->my : g->E)

/* gather/scatter between the flattened grid and compact face buffers */
static void ccopy(int n, double *dst, const double *src, int inc)
{
    int i;
    if (inc == 1) { memcpy(dst, src, (size_t)n * sizeof(double)); return; }
    for (i = 0; i < n; i++) dst[i] = src[(size_t)i * inc];
}

/* p, ke, invd, velocity.  vel = mom .* (1/d) elementwise: the SAME
 * scalar field feeds all three component slots of the 1/d view, each
 * read with stride 1.  (A stride of 0 repeats element 0 forever - that
 * is scalar broadcast, see boundaries() for the real use.)  3mom_ke
 * closes E = ke + p/(gamma-1) in one pass; axpy/scal finish. */
static void primitives(grid *g)
{
    int i, n = g->n;
    v3d m = v3d_wrap(g->mx, g->my, g->mz, n);
    v3d inv3 = v3d_wrap(g->invd, g->invd, g->invd, n);   /* stride 1!   */
    v3d v = v3d_wrap(g->vx, g->vy, g->vz, n);
    v3d km = v3d_wrap(g->km, g->km + g->n, g->km + 2 * g->n, n);
    for (i = 0; i < n; i++)
        g->invd[i] = 1.0 / (g->d[i] > SMALL ? g->d[i] : SMALL);
    v3had(m, inv3, v);                                 /* v = m.*(1/d)  */
    v3mom_ke(g->d, v, km, g->ke);                    /* ke = .5*d*(v.v) */
    memcpy(g->p, g->E, (size_t)n * sizeof(double));
    cblas_daxpy(n, -1.0, g->ke, 1, g->p, 1);
    cblas_dscal(n, GAMMA - 1.0, g->p, 1);            /* p=(g-1)(E-ke)   */
    v3sqr(v, g->vsq);
}

/* CFL bound (plus explicit-diffusion bound) by plain reduction loop:
 * global reductions are deliberately not kernels. */
static void time_step(grid *g, double cfl, double av)
{
    int i, j;
    double rmax = 0.0, dt;
    for (j = 1; j <= g->ny; j++)
        for (i = 1; i <= g->nx; i++) {
            int k = i + j * g->nxg;
            double c = sqrt(GAMMA * g->p[k] * g->invd[k]);
            double r = (fabs(g->vx[k]) + c) / g->dx
                     + (fabs(g->vy[k]) + c) / g->dy;
            if (r > rmax) rmax = r;
        }
    dt = cfl / (rmax > 0 ? rmax : 1.0);
    if (av > 0.0) {
        double ltd = 0.2 * g->dx * g->dx / av;
        if (ltd < dt) dt = ltd;
    }
    g->dt = dt;
}

/* x: zero-gradient copies; y: mirror d, E and reflect the momentum
 * with v3refl (a=2, unit normal broadcast, eps=0) = specular wall. */
static void boundaries(grid *g)
{
    int j, k;
    double zero = 0.0, one = 1.0;
    for (k = 0; k < 4; k++) {
        double *fld = UC(k);
        for (j = 0; j < g->nyg; j++) {
            double *r = AT(fld, j);
            r[0] = r[1];
            r[g->nxg - 1] = r[g->nxg - 2];
        }
    }
    for (k = 0; k < 2; k++) {                        /* bottom, top     */
        int jt = k ? g->nyg - 1 : 0, jc = k ? g->nyg - 2 : 1;
        memcpy(AT(g->d, jt), AT(g->d, jc), (size_t)g->nxg * sizeof(double));
        memcpy(AT(g->E, jt), AT(g->E, jc), (size_t)g->nxg * sizeof(double));
        {
            v3d m = v3d_wrap(AT(g->mx, jc), AT(g->my, jc), AT(g->mz, jc),
                             g->nxg);
            v3d nn = v3d_wrap(&zero, &one, &zero, g->nxg);
            v3d q = v3d_wrap(AT(g->mx, jt), AT(g->my, jt), AT(g->mz, jt),
                             g->nxg);
            nn.incx = 0; nn.incy = 0; nn.incz = 0;   /* bcast normal    */
            v3refl(m, nn, 2.0, 0.0, q);
        }
    }
}

/* U += -av*dt*lap5(U).  lap5 returns alpha*(sum4 - 4u) = -Laplacian,
 * hence the minus; jx=1, jy=nxg are the grid strides themselves.
 * Window = rows 1..nyg-2: every stencil read is in bounds.  Cells in
 * the ghost *columns* wrap around to the neighbouring row and get
 * nonsense values there - harmless, boundaries() rewrites all ghost
 * columns right after, and interior cells only ever read interior
 * neighbours. */
static void diffuse(grid *g, double av)
{
    int k;
    int nw = g->n - 2 * g->nxg;                        /* window cells  */
    for (k = 0; k < 4; k++) {
        cblas_dlap5(nw, 1.0, UC(k) + g->nxg, 1, 1.0, (double)g->nxg,
                    g->lap + g->nxg, 1);
        cblas_daxpy(nw, -av * g->dt, g->lap + g->nxg, 1, UC(k) + g->nxg, 1);
    }
}

/* Rusanov fluxes for nf consecutive faces of one grid line starting at
 * flat offset `off`, walking the grid with stride `str` (1 = x row,
 * nxg = y column); is_y picks the normal direction. */
static void face_fluxes(grid *g, int nf, size_t off, int str, int is_y)
{
    double *un = is_y ? g->vy : g->vx, *ut = is_y ? g->vx : g->vy;
    int i, k;
    /* gather left/right states into compact buffers */
    for (k = 0; k < 4; k++) {
        ccopy(nf, g->L[k], UC(k) + off, str);
        ccopy(nf, g->R[k], UC(k) + off + str, str);
    }
    ccopy(nf, g->vnL, un + off, str); ccopy(nf, g->vnR, un + off + str, str);
    ccopy(nf, g->vtL, ut + off, str); ccopy(nf, g->vtR, ut + off + str, str);
    ccopy(nf, g->pL, g->p + off, str); ccopy(nf, g->pR, g->p + off + str, str);
    /* local max wave speed per face (physics, not BLAS) */
    for (i = 0; i < nf; i++) {
        double cl = sqrt(GAMMA * g->pL[i] * (1.0 / (g->L[0][i] + SMALL)));
        double cr = sqrt(GAMMA * g->pR[i] * (1.0 / (g->R[0][i] + SMALL)));
        double sl = fabs(g->vnL[i]) + cl, sr = fabs(g->vnR[i]) + cr;
        g->s[i] = sl > sr ? sl : sr;
    }
    /* F_k = 0.5(q_L v_L + q_R v_R) [+ n_k avg(p)] - 0.5 s (q_R - q_L);
     * products via v1xypa (r = q.*t), sums via axpy/scal */
    for (k = 0; k < 4; k++) {
        double *F = g->F[k];
        if (k == 3) {                      /* energy: flux of (E+p)     */
            memcpy(g->t1, g->L[3], (size_t)nf * sizeof(double));
            cblas_daxpy(nf, 1.0, g->pL, 1, g->t1, 1);          /* Ep_L   */
            memcpy(g->t2, g->R[3], (size_t)nf * sizeof(double));
            cblas_daxpy(nf, 1.0, g->pR, 1, g->t2, 1);          /* Ep_R   */
            v1xypa(g->t1, g->vnL, 0.0, F, nf);
            v1xypa(g->t2, g->vnR, 0.0, g->t1, nf);
        } else if (k == 0) {               /* mass: flux of d           */
            v1xypa(g->L[0], g->vnL, 0.0, F, nf);
            v1xypa(g->R[0], g->vnR, 0.0, g->t1, nf);
        } else {                           /* momentum m_k: m_k*v_n ... */
            v1xypa(g->L[k], g->vnL, 0.0, F, nf);
            v1xypa(g->R[k], g->vnR, 0.0, g->t1, nf);
            if (k - 1 == is_y) {           /* ... + n_k * avg(p)        */
                memcpy(g->t2, g->pL, (size_t)nf * sizeof(double));
                cblas_daxpy(nf, 1.0, g->pR, 1, g->t2, 1);
                cblas_dscal(nf, 0.5, g->t2, 1);
            }
        }
        cblas_daxpy(nf, 1.0, g->t1, 1, F, 1);
        cblas_dscal(nf, 0.5, F, 1);
        if (k >= 1 && k - 1 == is_y)       /* pressure term (see above) */
            cblas_daxpy(nf, 1.0, g->t2, 1, F, 1);
        /* dissipation */
        memcpy(g->du, g->R[k], (size_t)nf * sizeof(double));
        cblas_daxpy(nf, -1.0, g->L[k], 1, g->du, 1);
        cblas_dscal(nf, -0.5, g->du, 1);
        v1xypa(g->s, g->du, 0.0, g->t1, nf);
        cblas_daxpy(nf, 1.0, g->t1, 1, F, 1);
    }
}

/* U[cell+str] -= dt/h * (F[i+1] - F[i]) for the nf-1 interior cells */
static void apply_line(grid *g, int nf, size_t off, int str, double h)
{
    int k, n = nf - 1;
    for (k = 0; k < 4; k++) {
        double *F = g->F[k], *U = UC(k) + off + str;
        memcpy(g->t1, F + 1, (size_t)n * sizeof(double));
        cblas_daxpy(n, -1.0, F, 1, g->t1, 1);
        cblas_dscal(n, -g->dt / h, g->t1, 1);
        cblas_daxpy(n, 1.0, g->t1, 1, U, str);
    }
}

static void report(grid *g, double t, int step)
{
    int i, j, bad = 0;
    double pmin = 1e30, mass = 0, momx = 0, mach = 0, m2;
    for (j = 1; j <= g->ny; j++)
        for (i = 1; i <= g->nx; i++) {
            int k = i + j * g->nxg;
            if (!(g->p[k] > 0.0) || !isfinite(g->E[k])) bad++;
            if (g->p[k] < pmin) pmin = g->p[k];
            mass += g->d[k];
            momx += g->mx[k];
            m2 = g->vsq[k] / (GAMMA * g->p[k] * g->invd[k]);
            if (m2 > mach) mach = m2;
        }
    printf("t=%7.4f step=%5d  minp=%.4e  mass=%12.8f  momx=%12.8f"
           "  M=%.3f%s\n", t, step, pmin, mass * g->dx * g->dy,
           momx * g->dx * g->dy, sqrt(mach),
           bad ? "  *** BAD CELLS ***" : "");
    fflush(stdout);
    if (bad) exit(1);
}

static void write_vtk(grid *g, double t)
{
    FILE *fp = fopen("sod2d.vtk", "w");
    int i, j;
    if (!fp) return;
    fprintf(fp, "# vtk DataFile Version 3.0\nsod2d t=%g\n"
            "ASCII\nDATASET STRUCTURED_POINTS\n"
            "DIMENSIONS %d %d 1\nORIGIN 0 0 0\nSPACING %g %g 1\n"
            "POINT_DATA %d\n", t, g->nx, g->ny, g->dx, g->dy,
            g->nx * g->ny);
    fprintf(fp, "SCALARS density double 1\nLOOKUP_TABLE default\n");
    for (j = 1; j <= g->ny; j++)
        for (i = 1; i <= g->nx; i++) fprintf(fp, "%.6f\n", AT(g->d, j)[i]);
    fprintf(fp, "SCALARS pressure double 1\nLOOKUP_TABLE default\n");
    for (j = 1; j <= g->ny; j++)
        for (i = 1; i <= g->nx; i++) fprintf(fp, "%.6f\n", AT(g->p, j)[i]);
    fprintf(fp, "VECTORS momentum double\n");
    for (j = 1; j <= g->ny; j++)
        for (i = 1; i <= g->nx; i++)
            fprintf(fp, "%.6f %.6f 0\n", AT(g->mx, j)[i], AT(g->my, j)[i]);
    fclose(fp);
}

int main(int argc, char **argv)
{
    grid g;
    int nx = 96, ny = 48, out = 50, step = 0, i;
    double tend = 0.2, cfl = 0.45, av = 0.02, t = 0.0;

    for (i = 1; i + 1 < argc; i += 2) {
        if      (!strcmp(argv[i], "-nx"))   nx   = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "-ny"))   ny   = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "-tend")) tend = atof(argv[i + 1]);
        else if (!strcmp(argv[i], "-cfl"))  cfl  = atof(argv[i + 1]);
        else if (!strcmp(argv[i], "-av"))   av   = atof(argv[i + 1]);
        else if (!strcmp(argv[i], "-out"))  out  = atoi(argv[i + 1]);
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }
    alloc_grid(&g, nx, ny);

    for (int j = 1; j <= g.ny; j++)                  /* Sod initial data */
        for (i = 1; i <= g.nx; i++) {
            int k = i + j * g.nxg, left = (i - 0.5) * g.dx < 0.5;
            g.d[k] = left ? 1.0 : 0.125;
            g.p[k] = left ? 1.0 : 0.1;
            g.E[k] = g.p[k] / (GAMMA - 1.0);
        }
    boundaries(&g);

    while (t < tend) {
        primitives(&g);
        time_step(&g, cfl, av);
        if (av > 0.0) diffuse(&g, av);
        for (int j = 0; j < g.nyg; j++) {            /* x: stride 1     */
            face_fluxes(&g, g.nxg - 1, (size_t)j * g.nxg, 1, 0);
            apply_line(&g, g.nxg - 1, (size_t)j * g.nxg, 1, g.dx);
        }
        for (i = 0; i < g.nxg; i++) {                /* y: stride nxg   */
            face_fluxes(&g, g.nyg - 1, (size_t)i, g.nxg, 1);
            apply_line(&g, g.nyg - 1, (size_t)i, g.nxg, g.dy);
        }
        boundaries(&g);
        t += g.dt; step++;
        if (step % out == 0) report(&g, t, step);
    }
    report(&g, t, step);
    write_vtk(&g, t);
    printf("sod2d: %d steps to t=%g on %dx%d; wrote sod2d.vtk\n",
           step, t, nx, ny);
    return 0;
}
