/* mhd_convert.c - see mhd_convert.h for the contract.
 * Division of labour is deliberate: everything that is physically a
 * vector field goes through a v3blas call (v3sqr for the norms, 1xypa
 * for field-times-field assembly); only genuinely scalar-field algebra
 * remains as one plain loop per output scalar field.  Thermodynamics
 * goes entirely through the EOS closure pair below.
 */
#include "mhd_convert.h"

/* P(rho, e): stiffened gas, ideal gas = pinf 0 */
double mhd_eos_p(double rho, double e, double gamma, double pinf)
{
  return (gamma - 1.0) * rho * e - gamma * pinf;
}

/* Q(rho, p): the inverse closure (a real-gas EOS solves it numerically) */
double mhd_eos_e(double rho, double p, double gamma, double pinf)
{
  return (p + gamma * pinf) / ((gamma - 1.0) * rho);
}

/* mom = rho*v,  E = rho*e(rho,pres) + 0.5*(rho*v.v + B.B) */
void mhd_prim2cons(const double *rho, const double *pres, double gamma, double pinf,
                   v3d v, v3d B, v3d mom, double *E)
{
  blasint i, n = v.n;
  double v2[n], B2[n];                  /* scratch energy fields */

  v3sqr_d(v, v2);                       /* v2 = v.v (Hermitian, real) */
  v3sqr_d(B, B2);                       /* B2 = B.B                  */
  v1xypa_d(rho, v.x, 0.0, mom.x, n);    /* mom = rho*v + 0,   */
  v1xypa_d(rho, v.y, 0.0, mom.y, n);    /* component by component */
  v1xypa_d(rho, v.z, 0.0, mom.z, n);

  for (i = 0; i < n; i++) {
    double e = mhd_eos_e(rho[i], pres[i], gamma, pinf);  /* specific */
    E[i] = rho[i] * e + 0.5 * (rho[i] * v2[i] + B2[i]);
  }
}

/* v = mom/rho,  pres = (gamma-1)*rho*e - gamma*pinf with e specific */
void mhd_cons2prim(const double *rho, const double *E, double gamma, double pinf,
                   v3d mom, v3d B, double *pres, v3d v)
{
  blasint i, n = v.n;
  double irho[n], v2[n], B2[n];

  for (i = 0; i < n; i++)
    irho[i] = 1.0 / rho[i];
  v1xypa_d(mom.x, irho, 0.0, v.x, n);   /* v = irho*mom + 0, */
  v1xypa_d(mom.y, irho, 0.0, v.y, n);   /* component by component */
  v1xypa_d(mom.z, irho, 0.0, v.z, n);

  v3sqr_d(v, v2);                       /* v2 = v.v of the NEW v */
  v3sqr_d(B, B2);                       /* B2 = B.B             */

  for (i = 0; i < n; i++) {
    double e = (E[i] - 0.5 * (rho[i] * v2[i] + B2[i])) * irho[i];  /* specific */
    pres[i] = mhd_eos_p(rho[i], e, gamma, pinf);
  }
}