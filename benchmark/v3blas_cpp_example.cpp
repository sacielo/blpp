/* v3blas_cpp_example.cpp - minimal tutorial for the v3blas.hpp C++
 * front-end: templates resolve the precision from the argument types,
 * 3-vectors are three arrays wrapped once in a view struct.
 *
 * One damped half-step of a particle: drag acceleration + kinetic energy.
 *
 * Build:
 *   g++ -O2 -std=c++11 -I ../pkgs/openblas/include/openblas \
 *       v3blas_cpp_example.cpp -L ../pkgs/openblas/lib -lopenblas -lm \
 *       -o v3blas_cpp_example
 */
#include "v3blas.hpp"
#include <cstdio>

int main()
{
    const int n = 4;
    double v1[n] = { 1, 2, 3, 4 };       /* velocity components:       */
    double v2[n] = { 0, 0, 0, 0 };       /* one array per component    */
    double v3[n] = { 3, 2, 1, 0 };
    double rho[n] = { 2, 2, 2, 2 };      /* density                    */
    double a1[n], a2[n], a3[n], ke[n];

    v3d v = v3d_wrap(v1, v2, v3, n);     /* view: 3 pointers + n, by value */

    v3drag(v, 0.5, 1e-12, a1, a2, a3);   /* a = 0.5*v*sqrt(v.v+eps)      */
    v3mom_ke(rho, v, a1, a2, a3, ke);    /* reuse a[] as m, k = .5*rho*v2*/

    printf("kinetic:");
    for (int i = 0; i < n; i++) printf(" %.1f", ke[i]);
    printf("  (expected 10 8 10 16)\n");
    return 0;
}
