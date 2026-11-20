/* mhd_convert.h - ideal-MHD primitive <-> conservative conversion (3D).
 *
 * The 8 physical variables, in v3blas notation:
 *   primitive set:    rho, pres (scalar fields),  v, B (3-vector fields)
 *   conserved set:    rho, E    (scalar fields),  mom, B (3-vector fields)
 *
 *     mom = rho*v                       (vector: recombine -> bilinear)
 *     E   = pres/(gamma-1) + 0.5*(rho*v.v + B.B)
 *
 * Thermodynamics is NOT inlined here - that is the point.  Both directions
 * delegate the p <-> e relation to the two-way EOS closure pair
 *     p = P(rho, e),   e = Q(rho, p)          (e = SPECIFIC internal energy)
 * just as a production compressible CFD code does, because real matter is
 * not ideal gas and Q is not always analytic (real-gas/tabulated EOS put a
 * Newton solve behind exactly these two entry points).  This file ships the
 * stiffened-gas pair (ideal gas is the pinf = 0 case).
 *
 * rho and B appear in both state sets IDENTICALLY (the mass density is
 * rho; B is already a conservative variable), so each takes exactly one
 * argument - no shadow copies. What is physically a scalar is a plain
 * unit-stride double array; what is physically a vector is one v3d view.
 * Length comes from the views (.n), as everywhere in v3blas. Outputs may
 * alias inputs unless noted.  Magnetic energy is B.B/2 (mu_0 = 1, the
 * standard nondimensional MHD scaling).
 */
#ifndef MHD_CONVERT_H
#define MHD_CONVERT_H

#include "v3blas.h"

/* the two-way EOS closure pair (stiffened gas; ideal gas = pinf 0) */
double mhd_eos_p(double rho, double e, double gamma, double pinf);  /* P(rho,e) */
double mhd_eos_e(double rho, double p, double gamma, double pinf);  /* Q(rho,p) */

/* prim -> cons: fills mom, E (rho and B pass through, untouched) */
void mhd_prim2cons(const double *rho, const double *pres, double gamma, double pinf,
                   v3d v, v3d B, v3d mom, double *E);

/* cons -> prim: fills pres, v (rho and B pass through, untouched) */
void mhd_cons2prim(const double *rho, const double *E, double gamma, double pinf,
                   v3d mom, v3d B, double *pres, v3d v);

#endif /* MHD_CONVERT_H */