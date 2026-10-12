/*
 * cblas_example.c - minimal tutorial for the physics extension kernels
 * as plain cblas_* calls: you pass every pointer and every stride.
 *
 * One explicit-Euler step of 1D heat conduction on a padded array with
 * cblas_slap5 (the 5-point Laplacian), then one cblas_s3cross.  The
 * padded layout is the whole trick: u[16] is the first physical point,
 * the stencil neighbours live at fixed element offsets jx = 3, jy = 7
 * (wide on purpose, so nothing looks accidental).
 *
 * Build (see CMakeLists.txt):
 *   gcc -O2 -I <prefix>/include/openblas cblas_example.c \
 *       -L <prefix>/lib -lopenblas -lm -o cblas_example
 */
#include "cblas.h"
#include <stdio.h>

#define NP   9                 /* physical points */
#define PAD  16                /* ghost cells on each side */

int main(void)
{
    static float u[PAD + NP + PAD];            /* [ghost | physical | ghost] */
    float lap[NP], heat[NP];
    float x1[NP], x2[NP], x3[NP], y1[NP], y2[NP], y3[NP];
    float w1[NP], w2[NP], w3[NP];
    int i;

    u[PAD + NP / 2] = 1.0f;                    /* one hot cell, rest is zero */

    /* lap = 0.25 * (u(i-3)+u(i+3)+u(i-7)+u(i+7) - 4*u(i)) */
    cblas_slap5(NP, 0.25f, u + PAD, 1, 3.0f, 7.0f, lap, 1);

    /* one diffusive step: heat = u + dt * lap, dt = 0.1 (axpy in place) */
    for (i = 0; i < NP; i++) heat[i] = u[PAD + i];
    cblas_saxpy(NP, 0.1f, lap, 1, heat, 1);

    printf("heat after one slap5 step:");
    for (i = 0; i < NP; i++) printf(" %.4f", heat[i]);
    printf("\n");

    /* 3-vectors are three plain arrays: x = (1,2,3), y = (i,1,0) */
    for (i = 0; i < NP; i++) {
        x1[i] = 1.0f; x2[i] = 2.0f; x3[i] = 3.0f;
        y1[i] = (float)i; y2[i] = 1.0f; y3[i] = 0.0f;
    }
    cblas_s3cross(NP, x1, 1, x2, 1, x3, 1, y1, 1, y2, 1, y3, 1,
                  w1, 1, w2, 1, w3, 1);
    /* w = x ^ y = (2*0-3*1, 3*i-1*0, 1*1-2*i) = (-3, 3i, 1-2i) */
    printf("3cross: w(0) = (%.0f, %.0f, %.0f)\n", w1[0], w2[0], w3[0]);
    return 0;
}
