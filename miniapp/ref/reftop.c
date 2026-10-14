/* reftop.c — independent reference for tops3d body 0. DO NOT refactor,
 * DO NOT share headers or helpers with tops3d.c, DO NOT call BLAS.
 *
 * Its whole value is that it re-derives the same physics a second time
 * from scratch — plain loops, hand cofactor inverse, hand cross product
 * — so agreement (~1e-14) pins the batched pipeline in tops3d: the
 * strided SoA->AoS pack, the dsbmv band fill, the v3cross Euler term and
 * the RK4 staging jointly. Body 0 is a FREE top, whose body-frame L
 * dynamics is autonomous (no attitude needed), and its initial data are
 * fixed constants replicated exactly in tops3d's init.
 *
 * Build & run:  cc -O2 miniapp/ref/reftop.c -o /tmp/reftop && /tmp/reftop
 * Compare against:  ./miniapp/build/tops3d | grep L_final   (defaults)
 */
#include <stdio.h>

int main(void)
{
    double I[3][3] = { { 2.0, 0.3, 0.1 }, { 0.3, 1.0, 0.2 },
                       { 0.1, 0.2, 0.5 } };
    /* hand inverse (cofactor/det), independently written */
    double det = I[0][0]*(I[1][1]*I[2][2] - I[1][2]*I[2][1])
               - I[0][1]*(I[0][1]*I[2][2] - I[1][2]*I[0][2])
               + I[0][2]*(I[0][1]*I[1][2] - I[1][1]*I[0][2]);
    double B[3][3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            int r1 = (i + 1) % 3, r2 = (i + 2) % 3;
            int c1 = (j + 1) % 3, c2 = (j + 2) % 3;
            /* cyclic (i+1,i+2)%3 minors already carry the cofactor
             * sign: NO extra (-1)^(i+j) parity factor */
            double cof = I[r1][c1]*I[r2][c2] - I[r1][c2]*I[r2][c1];
            B[j][i] = cof / det;
        }
    double L[3] = { 1.0, 0.6, 0.4 };
    double dt = 2e-3;
    for (int step = 0; step < 250; step++) {
        double k[4][3];
        for (int s = 0; s < 4; s++) {
            double h = (s == 0) ? 0.0 : (s < 3 ? 0.5*dt : dt);
            double Ls[3];
            for (int i = 0; i < 3; i++)
                Ls[i] = L[i] + h * (s == 0 ? 0.0 : k[s - 1][i]);
            double om[3];
            for (int i = 0; i < 3; i++)
                om[i] = B[i][0]*Ls[0] + B[i][1]*Ls[1] + B[i][2]*Ls[2];
            /* dL/dt = L x om, by hand */
            k[s][0] = Ls[1]*om[2] - Ls[2]*om[1];
            k[s][1] = Ls[2]*om[0] - Ls[0]*om[2];
            k[s][2] = Ls[0]*om[1] - Ls[1]*om[0];
        }
        for (int i = 0; i < 3; i++)
            L[i] += dt/6.0 * (k[0][i] + 2*k[1][i] + 2*k[2][i] + k[3][i]);
    }
    printf("ref: %.14f %.14f %.14f\n", L[0], L[1], L[2]);
    return 0;
}
