/* tops3d.c — N rigid tops: attitude dynamics as the meeting point of
 * physics kernels (BLAS-1 extension layer) and banded BLAS-2.
 *
 * What it simulates
 * -----------------
 * nT independent rigid bodies ("tops"). Body 0 is a FIXED free asymmetric
 * top (fixed data, so the run is reproducible against a hand-written
 * reference); body 1 is a HEAVY top pinned at a point, spun fast — it
 * precesses at the textbook rate. The rest are free asymmetric tops with
 * random data, here to make the invariants statistically meaningful.
 *
 * Free tops are one of the last problems in mechanics solved exactly
 * (Euler, 1758; elliptic functions). Their exact invariants — energy and
 * |L|^2 — are this app's correctness gate, the way the analytic Sod star
 * state gates sod2d.
 *
 * The physics, and why the frames matter
 * --------------------------------------
 * Let the body carry its own orthonormal frame {e1,e2,e3} rigidly (the
 * principal-ish frame below need NOT be principal — see "The banded
 * batch" for why that is a feature). Quantities marked B are expressed
 * in body axes, W in world (inertial) axes.
 *
 *   Euler's equation (BODY frame):
 *       dL/dt|B = tau_B + L_B x omega_B                      ... (1)
 *   The beauty: in the body frame the inertia tensor I is CONSTANT
 *   (the body carries its frame), so omega_B = I^-1 L_B is a fixed
 *   linear map and (1) is an autonomous ODE in L_B alone. In the world
 *   frame I(t) rotates every instant — hopeless to batch. This is why
 *   every serious rigid-body code lives in the body frame and carries
 *   the attitude separately.
 *
 *   Attitude: a unit quaternion q = (qs, qv), convention: q rotates
 *   BODY vectors into the WORLD, v_W = R(q) v_B. With omega expressed
 *   in BODY axes the kinematics is the RIGHT isochoric product:
 *       dq/dt = 1/2 q x (0, omega_B)
 *       qs' = -1/2 qv . omega                                  ... (2)
 *       qv' =  1/2 ( qs * omega + qv x omega )                 ... (3)
 *   (Get the left/right product wrong and precession runs backwards —
 *   the heavy top's precession direction is the experiment that tells
 *   you; the measured-vs-analytic report below is that experiment.)
 *
 *   Heavy top, pinned at O, center of mass at r_cm = (0,0,l) in BODY
 *   axes (constant there!), gravity g_W = (0,0,-g) in WORLD axes:
 *       tau_B = r_cm x ( m * R(q)^T g_W )                      ... (4)
 *   Note R^T: we must pull the world force INTO body axes before the
 *   cross product — (1) only accepts body-frame torques. Mixing frames
 *   inside one vector equation is the classic rigid-body bug; here the
 *   frames are spelled out at every call site.
 *
 *   Fast-top precession (what we check against): a top spinning with
 *   body-axis spin omega3, torque arm m*g*l, precesses about vertical at
 *       Omega_prec = m g l / (I3 omega3)                       ... (5)
 *   (leading order, fast spin). Exact to O(1/omega3).
 *
 * The banded batch — where Level-2 enters, honestly
 * -------------------------------------------------
 * Per body we need two small matrix-vector products, nT times per
 * right-hand-side evaluation:
 *     omega_B[b] = Iinv[b] (3x3 symmetric) L_B[b]
 *     g_B[b]     = R^T[b]  (3x3 general)     g_W[b]
 * Storing the batch as a block-diagonal BIG matrix and calling one L2
 * op is not a trick, it is the natural layout:
 *   - a block-diagonal matrix of 3x3 symmetric blocks IS a symmetric
 *     banded matrix with KD = 2 (block-boundary band slots simply hold
 *     zero); storage cost: (KD+1)*3nT = exactly one 3x3 per body.
 *   - likewise a batch of general 3x3 is a general band, KL = KU = 2.
 * One dsbmv/dgbmv call then streams the whole batch: contiguous memory,
 * one pass, bandwidth-bound — which for these byte-per-flop ratios is
 * precisely the regime where BLAS-2 earns its keep vs per-body loops.
 *
 * Data layout, and the bridge
 * ---------------------------
 * Element-wise physics wants SoA fields (this API's grammar): Lx[b],
 * Ly[b], Lz[b] each contiguous. The banded batch wants AoS: the L2
 * vector is [L0x L0y L0z L1x L1y L1z ...], body-major. The bridge is a
 * 3 x nT mini-transpose, and BLAS-1 does it with strided copies:
 *     dcopy(nT, Lx, 1, La + 0, 3)      (row -> every 3rd slot)
 * SoA<->AoS bridges are the most common plumbing in real BLAS codes;
 * here it is explicit rather than hidden in a struct loop.
 *
 * Kernel map (the point of the mini-app)
 * --------------------------------------
 *   omega = I^-1 L batch        cblas_dsbmv   KD=2 ld=3      <-- L2
 *   g_B = R^T g_W batch         cblas_dgbmv   KL=KU=2 ld=5   <-- L2
 *   SoA <-> AoS                 cblas_dcopy   stride 3       <-- L1 bridge
 *   L x omega, qv x omega       v3cross / v3crossscal
 *   r_cm x m g_B   (4)          v3cross + v3had (mass is a scalar
 *                               FIELD: same array in all three slots
 *                               with stride 1 — NOT a 0-stride
 *                               broadcast; see miniapp/README)
 *   qv . omega, w . L            v3dot
 *   qs * omega (scalar field!)  v1xypa  (t-slot, stride 1)
 *   r_cm = (0,0,l) TRUE vector broadcast -> stride-0 view, the one
 *   legitimate inc=0.
 *   RK4 stages                  axpy/scal (plain BLAS baseline)
 * Global reductions (energy etc.) stay plain loops: kernels make no
 * promises about summation order.
 *
 * Validation contract
 * -------------------
 * 1. free tops: E = 1/2 omega.L and |L|^2 are EXACT invariants; RK4
 *    should hold them to ~1e-6 relative over the default run (printed).
 * 2. body 0 (fixed data) must agree with an independent hand-written
 *    reference (plain loops, own 3x3 inverse, same RK4) to ~1e-10.
 * 3. heavy top: measured precession rate vs eq. (5), within the fast-top
 *    approximation.
 * 4. VTK output: body 0 traces its POLHODE (the omega vector in body
 *    axes sweeps an exact cone-like curve — Euler's solution); body 1's
 *    world axis traces the precession cone. Wrong physics is visible.
 *
 * Build: cmake -S miniapp -B miniapp/build && cmake --build miniapp/build
 * Run:   ./miniapp/build/tops3d  [-nT 1024] [-tend 0.5] [-dt 2e-3] [-out 5]
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cblas.h"
#include "v3blas.h"

#define PI 3.14159265358979323846

static int nT = 1024;
static double tend = 0.5, dt = 2e-3;
static int outiv = 5;                /* report/trail interval (steps)  */
static double grav = 1.0, arm = 1.0; /* heavy top: g, and |r_cm| */

/* one LCG so every run — and the reference program — sees the same tops */
static unsigned int rngs = 12345u;
static double rnd(void) {
  rngs = rngs * 1103515245u + 12345u;
  return (double)(rngs >> 8) / 8388608.0; /* [0,1) */
}

/* ---- state: the fields (SoA, length nT). Body 0 = fixed check data;
 * body 1 = heavy fast top; bodies 2.. random free tops. ---- */
enum { YLX, YLY, YLZ, YQX, YQY, YQZ, YQS, NC };
static double *y[NC], *k1[NC], *k2[NC], *k3[NC], *k4[NC], *yst[NC];

/* static per-body data */
static double *mass;            /* scalar FIELD: 0 except heavy    */
static double *b13, *b23, *b33; /* Iinv z-column (report only)     */

/* banded batch stores (see "The banded batch" above) */
static double *bandI;             /* dsbmv upper band, ld = 3        */
static double *bandRT;            /* dgbmv band of R(q)^T, ld = 5    */
static double *La, *Wa, *Ga, *Fw; /* AoS: L batch, omega batch,
                                       g in body axes, world force     */

/* per-evaluation SoA scratch */
static double *omx, *omy, *omz, *gbx, *gby, *gbz, *mbx, *mby, *mbz;
static double *tx, *ty, *tz, *wx, *wy, *wz, *tau, *tau2, *tau3;
static double *dotb, *qs2, half, zero = 0.0;

/* trails for the VTK */
#define MAXPTS 4000
static double tr1[3 * MAXPTS], tr2[3 * MAXPTS];
static int np1 = 0, np2 = 0;

static void *dz(void) { return calloc((size_t)nT, sizeof(double)); }

/* pack SoA -> AoS (the 3xnT mini-transpose, three strided copies) */
static void pack3(double *A, double *x, double *yy, double *z) {
  cblas_dcopy(nT, x, 1, A + 0, 3);
  cblas_dcopy(nT, yy, 1, A + 1, 3);
  cblas_dcopy(nT, z, 1, A + 2, 3);
}
static void unpack3(double *A, double *x, double *yy, double *z) {
  cblas_dcopy(nT, A + 0, 3, x, 1);
  cblas_dcopy(nT, A + 1, 3, yy, 1);
  cblas_dcopy(nT, A + 2, 3, z, 1);
}

/* R(q)^T batched into the general band (KL=KU=2, ld=5):
 * slot(j-th column, row i) = (kl+ku + i - j) + j*5.
 * R maps body->world, so we store Rt = R^T to pull world vectors in.
 * Block-adjacent slots (|i-j|=1,2 across a boundary) must stay zero —
 * the band cannot "know" about block structure, we simply zero them. */
static void build_bandRT(const double *qx, const double *qy, const double *qz, const double *qs) {
  memset(bandRT, 0, sizeof(double) * (size_t)(5 * 3 * nT));
  for (int b = 0; b < nT; b++) {
    double a = qs[b], u = qx[b], v = qy[b], w = qz[b];
    double R[3][3] = {{1 - 2 * (v * v + w * w), 2 * (u * v - a * w), 2 * (u * w + a * v)}, {2 * (u * v + a * w), 1 - 2 * (u * u + w * w), 2 * (v * w - a * u)}, {2 * (u * w - a * v), 2 * (v * w + a * u), 1 - 2 * (u * u + v * v)}};
    double *B = bandRT + 15 * b; /* band column 3b: 5 rows each */
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) B[2 + i - j + 5 * j] = R[j][i]; /* Rt[i][j] = R[j][i]; slot = ku + i - j */
  }
}

/* Right-hand side of (1),(2),(3),(4). Reads state array sy[], writes
 * derivatives sk[]. Everything body-frame unless the name says _W. */
static void rhs(double *const *sy, double *const *sk) {
  v3d L = v3d_wrap(sy[YLX], sy[YLY], sy[YLZ], nT);
  v3d om = v3d_wrap(omx, omy, omz, nT);
  v3d QV = v3d_wrap(sy[YQX], sy[YQY], sy[YQZ], nT);

  /* --- omega_B = Iinv L_B for ALL bodies: one dsbmv over the block
     * diagonal. Upper band, KD=2, ld=3. The blocks are NOT diagonal
     * (random symmetric, full 3x3): this is why we call L2 and not
     * three scalings — asymmetric inertia is the interesting case. --- */
  pack3(La, sy[YLX], sy[YLY], sy[YLZ]);
  cblas_dsbmv(CblasColMajor, CblasUpper, 3 * nT, 2, 1.0, bandI, 3, La, 1, 0.0, Wa, 1);
  unpack3(Wa, omx, omy, omz);

  /* --- Euler term (1): dL/dt|B = tau_B + L x omega  (body x body ->
     * body). Output goes straight into the derivative fields. --- */
  v3cross(L, om, v3d_wrap(sk[YLX], sk[YLY], sk[YLZ], nT));

  /* --- heavy-top torque (4): g_W -> body axes via R^T batch =
     * one dgbmv, then body-frame cross with r_cm. --- */
  build_bandRT(sy[YQX], sy[YQY], sy[YQZ], sy[YQS]);
  /* Fw (built at init) is the per-body world force (0,0,-m g) in AoS;
     * bodies with m=0 contribute zeros, the batch does not care. */
  cblas_dgbmv(CblasColMajor, CblasNoTrans, 3 * nT, 3 * nT, 2, 2, 1.0, bandRT, 5, Fw, 1, 0.0, Ga, 1);
  unpack3(Ga, gbx, gby, gbz);
  /* mass is a scalar FIELD -> same array, all slots, stride 1
     * (sod2d's lesson: stride 0 here would freeze it at mass[0]) */
  v3d M = v3d_wrap(mass, mass, mass, nT);
  v3had(v3d_wrap(gbx, gby, gbz, nT), M, v3d_wrap(mbx, mby, mbz, nT));
  /* r_cm = (0,0,arm), the SAME for all bodies: a true vector
     * broadcast — the legitimate stride-0 view. */
  v3d rc = v3d_wrap(&zero, &zero, &arm, nT);
  rc.inc = 0;
  v3cross(rc, v3d_wrap(mbx, mby, mbz, nT), v3d_wrap(tau, tau2, tau3, nT));
  cblas_daxpy(nT, 1.0, tau, 1, sk[YLX], 1);
  cblas_daxpy(nT, 1.0, tau2, 1, sk[YLY], 1);
  cblas_daxpy(nT, 1.0, tau3, 1, sk[YLZ], 1);

  /* --- quaternion kinematics (2),(3) --- */
  /* qs' = -1/2 qv.omega */
  v3dot(QV, om, dotb);
  cblas_dcopy(nT, dotb, 1, sk[YQS], 1);
  cblas_dscal(nT, -0.5, sk[YQS], 1);
  /* qv' = 1/2(qs*omega) + 1/2(qv x omega):
     * the scalar times a 3-vector uses 1xypa's t-slot with the qs FIELD
     * at stride 1; the cross part is exactly v3crossscal's job. */
  cblas_d1xypa(nT, sy[YQS], 1, &half, 0, 0.0, qs2, 1); /* qs2 = qs/2 */
  v3crossscal(QV, om, half, v3d_wrap(wx, wy, wz, nT)); /* w=half*qv^om */
  cblas_d1xypa(nT, omx, 1, qs2, 1, 0.0, tx, 1);        /* tx =  qs2*omx    */
  cblas_d1xypa(nT, omy, 1, qs2, 1, 0.0, ty, 1);
  cblas_d1xypa(nT, omz, 1, qs2, 1, 0.0, tz, 1);
  cblas_daxpy(nT, 1.0, tx, 1, wx, 1);
  cblas_dcopy(nT, wx, 1, sk[YQX], 1);
  cblas_daxpy(nT, 1.0, ty, 1, wy, 1);
  cblas_dcopy(nT, wy, 1, sk[YQY], 1);
  cblas_daxpy(nT, 1.0, tz, 1, wz, 1);
  cblas_dcopy(nT, wz, 1, sk[YQZ], 1);
}

/* energy of one body: 1/2 omega.L (kinetic, frame-free scalar) */
static double energy_of(int b) {
  double *B = bandI + 3 * (3 * b); /* this body's band columns */
  double Lx = y[YLX][b], Ly = y[YLY][b], Lz = y[YLZ][b];
  /* upper band, ld 3: slot (i - j + KD) + 3j holds A[i,j], i <= j;
     * mirror for the lower triangle (band storage is column-major BY
     * DEFINITION, which is why the calls pass CblasColMajor) */
  double s[3][3];
  for (int j = 0; j < 3; j++)
    for (int i = 0; i < 3; i++) {
      double v = (i <= j) ? B[i - j + 2 + 3 * j] : B[j - i + 2 + 3 * i];
      s[i][j] = v;
    }
  double om_[3] = {s[0][0] * Lx + s[0][1] * Ly + s[0][2] * Lz, s[0][1] * Lx + s[1][1] * Ly + s[1][2] * Lz, s[0][2] * Lx + s[1][2] * Ly + s[2][2] * Lz};
  double bb = 0.5 * (om_[0] * Lx + om_[1] * Ly + om_[2] * Lz);
  if (b == 1) { /* heavy top: potential m g l * (height of CM) =
                     * m g l * (R e3)_z, the body z-axis in world */
    double u = y[YQX][b], v = y[YQY][b];
    bb += mass[b] * grav * arm * (1 - 2 * (u * u + v * v));
  }
  return bb;
}

int main(int ac, char **av) {
  for (int i = 1; i < NC + 4; i += 2)
    if (i + 1 >= ac) break;
    else if (!strcmp(av[i], "-nT"))
      nT = atoi(av[i + 1]);
    else if (!strcmp(av[i], "-tend"))
      tend = atof(av[i + 1]);
    else if (!strcmp(av[i], "-dt"))
      dt = atof(av[i + 1]);
    else if (!strcmp(av[i], "-out"))
      outiv = atoi(av[i + 1]);
  half = 0.5;

  /* allocate */
  for (int c = 0; c < NC; c++) {
    y[c] = dz();
    k1[c] = dz();
    k2[c] = dz();
    k3[c] = dz();
    k4[c] = dz();
    yst[c] = dz();
  }
  mass = dz();
  b13 = dz();
  b23 = dz();
  b33 = dz();
  La = calloc(3 * nT, sizeof(double));
  Wa = calloc(3 * nT, sizeof(double));
  Ga = calloc(3 * nT, sizeof(double));
  Fw = calloc(3 * nT, sizeof(double));
  bandI = calloc((size_t)(3 * 3 * nT), sizeof(double));
  bandRT = calloc((size_t)(5 * 3 * nT), sizeof(double));
  omx = dz();
  omy = dz();
  omz = dz();
  gbx = dz();
  gby = dz();
  gbz = dz();
  mbx = dz();
  mby = dz();
  mbz = dz();
  tx = dz();
  ty = dz();
  tz = dz();
  wx = dz();
  wy = dz();
  wz = dz();
  tau = dz();
  tau2 = dz();
  tau3 = dz();
  dotb = dz();
  qs2 = dz();

  /* ---- initial data ----
     * Body frame = "engineering" frame, NOT necessarily principal:
     * random symmetric I with real off-diagonal terms. Then omega and L
     * differ in direction, the dynamics gets interesting, and the dsbmv
     * batch is genuinely a symmetric solve, not three multiplications. */
  for (int b = 0; b < nT; b++) {
    double d[3], o[3], Ib[3][3], det, Bi[3][3];
    y[YQS][b] = 1.0; /* identity quaternion unless overridden:
                                * NOT optional — renorm divides by |q|,
                                * and a zero quaternion is a NaN faucet */
    if (b == 0) {    /* fixed check data (see reference pgm) */
      d[0] = 2.0;
      d[1] = 1.0;
      d[2] = 0.5;
      o[0] = 0.3;
      o[1] = 0.1;
      o[2] = 0.2;
      y[YLX][b] = 1.0;
      y[YLY][b] = 0.6;
      y[YLZ][b] = 0.4;
    } else if (b == 1) { /* heavy fast top: I3 small, big L3     */
      d[0] = 0.11;
      d[1] = 0.12;
      d[2] = 0.10;
      o[0] = 0.004;
      o[1] = 0.003;
      o[2] = 0.005;
      y[YLX][b] = 0.05;
      y[YLY][b] = 0.02;
      y[YLZ][b] = 2.0;
      mass[b] = 1.0;
      {
        double th = 0.15; /* tilt 0.15 rad about e1 */
        y[YQS][b] = cos(0.5 * th);
        y[YQX][b] = sin(0.5 * th);
      }
    } else { /* random free tops */
      d[0] = 0.4 + 1.8 * rnd();
      d[1] = 0.4 + 1.8 * rnd();
      d[2] = 0.4 + 1.8 * rnd();
      do {
        o[0] = 0.3 * (rnd() - 0.5);
        o[1] = 0.3 * (rnd() - 0.5);
        o[2] = 0.3 * (rnd() - 0.5);
      } while (fabs(o[0]) > fmin(d[0], d[1]) * 0.4 || fabs(o[1]) > fmin(d[1], d[2]) * 0.4 || fabs(o[2]) > fmin(d[0], d[2]) * 0.4);
      y[YLX][b] = 2 * rnd() - 1;
      y[YLY][b] = 2 * rnd() - 1;
      y[YLZ][b] = 2 * rnd() - 1;
    }
    Ib[0][0] = d[0];
    Ib[1][1] = d[1];
    Ib[2][2] = d[2];
    Ib[0][1] = Ib[1][0] = o[0]; /* xy */
    Ib[0][2] = Ib[2][0] = o[1]; /* xz */
    Ib[1][2] = Ib[2][1] = o[2]; /* yz */
    det = Ib[0][0] * (Ib[1][1] * Ib[2][2] - Ib[1][2] * Ib[1][2]) - Ib[0][1] * (Ib[0][1] * Ib[2][2] - Ib[1][2] * Ib[0][2]) + Ib[0][2] * (Ib[0][1] * Ib[1][2] - Ib[1][1] * Ib[0][2]);
    Bi[0][0] = (Ib[1][1] * Ib[2][2] - Ib[1][2] * Ib[2][1]) / det;
    Bi[1][1] = (Ib[0][0] * Ib[2][2] - Ib[0][2] * Ib[2][0]) / det;
    Bi[2][2] = (Ib[0][0] * Ib[1][1] - Ib[0][1] * Ib[1][0]) / det;
    Bi[0][1] = (Ib[2][1] * Ib[0][2] - Ib[0][1] * Ib[2][2]) / det;
    Bi[0][2] = (Ib[0][1] * Ib[1][2] - Ib[0][2] * Ib[1][1]) / det;
    Bi[1][2] = (Ib[0][2] * Ib[1][0] - Ib[0][0] * Ib[1][2]) / det;
    Bi[1][0] = Bi[0][1];
    Bi[2][0] = Bi[0][2];
    Bi[2][1] = Bi[1][2];
    /* park this body's symmetric block in the dsbmv UPPER band:
         * slot (i - j + KD) + j*ld, ld = KD+1 = 3, only i <= j filled */
    for (int j = 0; j < 3; j++)
      for (int i = 0; i <= j; i++) bandI[(i - j + 2) + 3 * (3 * b + j)] = Bi[i][j];
    b13[b] = Bi[0][2];
    b23[b] = Bi[1][2];
    b33[b] = Bi[2][2];
  }
  /* per-body world force, AoS: (0, 0, -m g)  (zero unless heavy) */
  for (int b = 0; b < nT; b++) Fw[3 * b + 2] = -mass[b] * grav;

  /* ---- initial invariants and spin, for the drift report ---- */
  double E0 = 0, L20 = 0, om3he;
  {
    pack3(La, y[YLX], y[YLY], y[YLZ]);
    cblas_dsbmv(CblasColMajor, CblasUpper, 3 * nT, 2, 1.0, bandI, 3, La, 1, 0.0, Wa, 1);
    unpack3(Wa, omx, omy, omz);
    for (int b = 0; b < nT; b++) {
      E0 += energy_of(b);
      if (b != 1) L20 += y[YLX][b] * y[YLX][b] + y[YLY][b] * y[YLY][b] + y[YLZ][b] * y[YLZ][b];
    }
    om3he = b13[1] * y[YLX][1] + b23[1] * y[YLY][1] + b33[1] * y[YLZ][1];
  }

  /* ---- integrate: classical RK4 over the 7-component state ----
     * (free top: RK4's tiny secular drift is exactly what the printed
     * invariant drift measures; heavy top: dt*omega ~ 0.04, ample) */
  double t = 0.0;
  int step = 0, outc = 0, init = 1;
  double prec = 0.0, phprev = 0.0;
  while (t < tend) {
    for (int s = 0; s < 4; s++) { /* stage weights */
      double h = (s == 0) ? 0.0 : (s < 3 ? 0.5 : 1.0);
      double **ks = (s == 0 ? k1 : s == 1 ? k2 : s == 2 ? k3 : k4);
      for (int c = 0; c < NC; c++) /* yst = y+h*dt*k */
        cblas_dcopy(nT, y[c], 1, yst[c], 1);
      if (s > 0) {
        double **kp = (s == 1 ? k1 : s == 2 ? k2 : k3);
        for (int c = 0; c < NC; c++) cblas_daxpy(nT, dt * h, kp[c], 1, yst[c], 1);
      }
      rhs(yst, ks);
    }
    for (int c = 0; c < NC; c++) { /* y += dt/6 (k1+2k2+2k3+k4) */
      cblas_daxpy(nT, 2.0, k2[c], 1, k1[c], 1);
      cblas_daxpy(nT, 2.0, k3[c], 1, k1[c], 1);
      cblas_daxpy(nT, 1.0, k4[c], 1, k1[c], 1);
      cblas_dscal(nT, dt / 6.0, k1[c], 1);
      cblas_daxpy(nT, 1.0, k1[c], 1, y[c], 1);
    }
    t += dt;
    step++;
    if (step % outiv == 0) {
      /* refresh omega at the NEW state (the scratch omx/.. still
             * hold the last stage evaluation's values) */
      pack3(La, y[YLX], y[YLY], y[YLZ]);
      cblas_dsbmv(CblasColMajor, CblasUpper, 3 * nT, 2, 1.0, bandI, 3, La, 1, 0.0, Wa, 1);
      unpack3(Wa, omx, omy, omz);
      /* renormalize quaternions (they are 4-vectors; this API
             * speaks 3-vectors, so: |q|^2 = qs^2 + qv.qv with a plain
             * loop for the per-body inverse norm — design boundary) */
      v3dot(v3d_wrap(y[YQX], y[YQY], y[YQZ], nT), v3d_wrap(y[YQX], y[YQY], y[YQZ], nT), dotb);
      for (int b = 0; b < nT; b++) {
        double nq = 1.0 / sqrt(dotb[b] + y[YQS][b] * y[YQS][b]);
        y[YQS][b] *= nq;
        y[YQX][b] *= nq;
        y[YQY][b] *= nq;
        y[YQZ][b] *= nq;
      }
      if (++outc % 5 == 0 || init) { /* trail samples */
        init = 0;
        if (np1 < MAXPTS) { /* body 0 polhode
                                                       * (omega in BODY
                                                       * axes: Euler's
                                                       * exact curve) */
          tr1[3 * np1] = omx[0];
          tr1[3 * np1 + 1] = omy[0];
          tr1[3 * np1 + 2] = omz[0];
          np1++;
        }
        if (np2 < MAXPTS) { /* heavy axis in
                     * WORLD: R e3 = column 3 of R(q): the precession
                     * cone, visibly conical if physics is right */
          double u = y[YQX][1], v = y[YQY][1], w = y[YQZ][1], a = y[YQS][1];
          tr2[3 * np2] = 2 * (u * w + a * v);
          tr2[3 * np2 + 1] = 2 * (v * w - a * u);
          tr2[3 * np2 + 2] = 1 - 2 * (u * u + v * v);
          np2++;
        }
        /* Precession is measured on L_hat in WORLD axes, NOT
                 * on the figure axis: the axis nutates about L at the
                 * fast rate ~L/I1 here, so its azimuth is mush; but
                 * |dL_hat/dt| = |tau|/L is slow by torque budget, so
                 * L_hat's azimuth is a clean precession signal.
                 * L_world = R(q) L_body in a plain per-body report
                 * loop (diagnostics do not need the batch). */
        double a = y[YQS][1], u = y[YQX][1], v = y[YQY][1], w = y[YQZ][1];
        double R[3][3] = {{1 - 2 * (v * v + w * w), 2 * (u * v - a * w), 2 * (u * w + a * v)}, {2 * (u * v + a * w), 1 - 2 * (u * u + w * w), 2 * (v * w - a * u)}, {2 * (u * w - a * v), 2 * (v * w + a * u), 1 - 2 * (u * u + v * v)}};
        double Lb[3] = {y[YLX][1], y[YLY][1], y[YLZ][1]};
        double lx = R[0][0] * Lb[0] + R[0][1] * Lb[1] + R[0][2] * Lb[2];
        double ly = R[1][0] * Lb[0] + R[1][1] * Lb[1] + R[1][2] * Lb[2];
        double ph = atan2(ly, lx); /* azimuth of L_hat_W */
        if (outc > 6) {
          double d = ph - phprev;
          while (d > PI) d -= 2 * PI;
          while (d < -PI) d += 2 * PI;
          prec += d;
        }
        phprev = ph;
      }
    }
  }

  /* ---- report ---- */
  double E = 0, L2 = 0;
  for (int b = 0; b < nT; b++) {
    E += energy_of(b);
    if (b != 1) L2 += y[YLX][b] * y[YLX][b] + y[YLY][b] * y[YLY][b] + y[YLZ][b] * y[YLZ][b];
  }
  double om3 = b13[1] * y[YLX][1] + b23[1] * y[YLY][1] + b33[1] * y[YLZ][1];
  double om3avg = 0.5 * (om3 + om3he);
  double Om_meas = prec / t;
  double Om_the = mass[1] * grav * arm / (0.1 * om3avg); /* I3 = 0.10 */
  printf("tops3d: %d steps, t=%.3f, %d tops\n", step, t, nT);
  printf("  energy   : rel drift  %+.3e   (exact invariant; RK4 holds)\n", (E - E0) / fabs(E0));
  printf("  |L|^2    : rel drift  %+.3e   (free tops; exact invariant)\n", (L2 - L20) / L20);
  printf("  precess  : measured %+.4f rad/s, fast-top theory %+.4f "
         "(mgl/I3 w3)\n",
         Om_meas, Om_the);
  printf("  body 0 (fixed data, cross-check vs hand reference):\n"
         "    L_final = %.15g %.15g %.15g\n",
         y[YLX][0], y[YLY][0], y[YLZ][0]);

  FILE *fp = fopen("tops3d.vtk", "w");
  fprintf(fp,
          "# vtk DataFile Version 3.0\ntops3d: polhode (body axes)"
          " + precession cone (world axes)\nASCII\n"
          "DATASET POLYDATA\nPOINTS %d double\n",
          np1 + np2);
  for (int i = 0; i < np1; i++) fprintf(fp, "%.6f %.6f %.6f\n", tr1[3 * i], tr1[3 * i + 1], tr1[3 * i + 2]);
  for (int i = 0; i < np2; i++) fprintf(fp, "%.6f %.6f %.6f\n", tr2[3 * i], tr2[3 * i + 1], tr2[3 * i + 2]);
  fprintf(fp, "LINES 2 %d\n%d", np1 + np2 + 2, np1);
  for (int i = 0; i < np1; i++) fprintf(fp, " %d", i);
  fprintf(fp, "\n%d", np2);
  for (int i = 0; i < np2; i++) fprintf(fp, " %d", np1 + i);
  fprintf(fp, "\n");
  fclose(fp);
  printf("  wrote tops3d.vtk: line 1 = body-0 polhode, "
         "line 2 = heavy-top cone\n");
  return 0;
}
