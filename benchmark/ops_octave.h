/*
 * ops_octave.h - Octave/MATLAB check block for the physics kernels.
 *
 * Printed ALWAYS (no flag), at d precision only: the other precisions
 * return early.  The point is that the test's whole stdout is then a
 * valid .m file:  ./cblas_test d > check.m && octave check.m
 * Every human-readable message in the tests is a % comment for this
 * reason; nothing else may be printed.
 *
 * Per output element, two lines, numbers inline everywhere:
 *     d3dot_save = 15             <- what the BLAS kernel returned
 *     d3dot = dot([1,2,3],[4,5,6]) <- recomputed from the inputs
 * NO trailing semicolon: Octave/MATLAB must echo both lines so the
 * printed pair can be compared by eye (a silent run proves nothing).
 * using Octave/MATLAB builtins (dot, cross), i.e. an INDEPENDENT
 * implementation - not a re-typed copy of our kernel.  The input
 * literals are expanded into every formula as printed numbers, so
 * there is no variable preamble; the only shared line is U=[...] for
 * the lap5/lap7 stencils (49 numbers, cited 9 times).  Compare by
 * eye, or diff in Octave.
 *
 * Caveat if c/z printing is ever enabled: dot() conjugates its first
 * argument while all our products are bilinear, so dot-family slots
 * would need dot(conj(A),B).  cross() is bilinear and safe as-is.
 *
 * Requires the xval()/cr machinery of the including test.
 * Used by cblas_test.c and v3blas_test.c only - never by benchmarks.
 */
#ifndef OPS_OCTAVE_H
#define OPS_OCTAVE_H

#include <ctype.h>
#include <string.h>

/* slot order matches the g[] arrays of the tests:
 *  0 axpy, 1 1xypa, 2 3dot, 3-5 3had, 6 3sqr, 7 1norm, 8-10 3cross,
 * 11-13 3crossscal, 14 3crossdot, 15 3crosssqr, 16-18 crossxy_crossxz.u,
 * 19-21 crossxy_crossxz.v, 22-24 crossxy_dotxz.u, 25 crossxy_dotxz.r,
 * 26 dotxy_dotxz.r, 27 dotxy_dotxz.q, 28-30 crosscross, 31 norm_unit.r,
 * 32-34 norm_unit.u, 35-37 refl, 38-40 exb, 41-43 drag, 44-46 mom,
 * 47 ke, 48 lap5, 49 lap7.
 * %s = precision prefix.  Scalars are baked in (a = 1.5, eps = 0.25;
 * 0.25+0.1i for 1xypa, 1.5+0.5i for axpy/crossscal).  Component slots
 * carry "(n)": the save line indexes the saved vector, the formula
 * line indexes the builtin result - both legal Octave and MATLAB. */
#define OV_VX   "[x1(1),x2(1),x3(1)]"
#define OV_VY   "[y1(1),y2(1),y3(1)]"
#define OV_VW   "[w1(1),w2(1),w3(1)]"
#define OV_DOT  "dot(" OV_VX "," OV_VY ")"
#define OV_SQR  "dot(" OV_VX "," OV_VX ")"
#define OV_XW   "dot(" OV_VX "," OV_VW ")"
#define OV_YY   "dot(" OV_VY "," OV_VY ")"
#define OV_CXY  "cross(" OV_VX "," OV_VY ")"

static const char *ov_names[50] = {
    "%saxpy", "%s1xypa", "%s3dot",
    "%s3had(1)", "%s3had(2)", "%s3had(3)",
    "%s3sqr", "%s1norm",
    "%s3cross(1)", "%s3cross(2)", "%s3cross(3)",
    "%s3crossscal(1)", "%s3crossscal(2)", "%s3crossscal(3)",
    "%s3crossdot", "%s3crosssqr",
    "%scrossxy_crossxz.u(1)", "%scrossxy_crossxz.u(2)", "%scrossxy_crossxz.u(3)",
    "%scrossxy_crossxz.v(1)", "%scrossxy_crossxz.v(2)", "%scrossxy_crossxz.v(3)",
    "%scrossxy_dotxz.u(1)", "%scrossxy_dotxz.u(2)", "%scrossxy_dotxz.u(3)",
    "%scrossxy_dotxz.r",
    "%sdotxy_dotxz.r", "%sdotxy_dotxz.q",
    "%s3crosscross(1)", "%s3crosscross(2)", "%s3crosscross(3)",
    "%s3norm_unit", "%s3norm_unit.u(1)", "%s3norm_unit.u(2)", "%s3norm_unit.u(3)",
    "%s3refl(1)", "%s3refl(2)", "%s3refl(3)",
    "%s3exb(1)", "%s3exb(2)", "%s3exb(3)",
    "%s3drag(1)", "%s3drag(2)", "%s3drag(3)",
    "%s3mom(1)", "%s3mom(2)", "%s3mom(3)",
    "%s3ke", "%slap5", "%slap7"
};

static const char *ov_expr_r[50] = {
    "1.5*xv(1)+yv(1)",                                       /* axpy */
    "q(1).*t(1)+0.25",                                       /* 1xypa */
    OV_DOT,                                                  /* 3dot */
    "x1(1).*y1(1)", "x2(1).*y2(1)", "x3(1).*y3(1)",          /* 3had */
    OV_SQR,                                                  /* 3sqr */
    "abs(q(1))",                                             /* 1norm */
    OV_CXY "(1)", OV_CXY "(2)", OV_CXY "(3)",          /* 3cross */
    "1.5*" OV_CXY "(1)", "1.5*" OV_CXY "(2)",
    "1.5*" OV_CXY "(3)",                                   /* crossscal */
    "dot(" OV_CXY "," OV_VW ")",                             /* crossdot */
    "dot(" OV_CXY "," OV_CXY ")",                            /* crosssqr */
    OV_CXY "(1)", OV_CXY "(2)", OV_CXY "(3)",               /* cxy.u */
    "cross(" OV_VX "," OV_VW ")(1)", "cross(" OV_VX "," OV_VW ")(2)",
    "cross(" OV_VX "," OV_VW ")(3)",                         /* cxy.v */
    OV_CXY "(1)", OV_CXY "(2)", OV_CXY "(3)",               /* cxdot.u */
    "dot(" OV_CXY "," OV_VW ")",                             /* cxdot.r */
    OV_DOT, OV_XW,                                           /* dxdx r,q */
    "cross(" OV_CXY "," OV_VW ")(1)",                        /* crosscross */
    "cross(" OV_CXY "," OV_VW ")(2)",
    "cross(" OV_CXY "," OV_VW ")(3)",
    "sqrt(" OV_SQR "+0.25)",                                 /* norm_unit r */
    "x1(1)/sqrt(" OV_SQR "+0.25)",
    "x2(1)/sqrt(" OV_SQR "+0.25)",
    "x3(1)/sqrt(" OV_SQR "+0.25)",
    "x1(1)-1.5*" OV_DOT "/(" OV_YY "+0.25)*y1(1)",           /* refl */
    "x2(1)-1.5*" OV_DOT "/(" OV_YY "+0.25)*y2(1)",
    "x3(1)-1.5*" OV_DOT "/(" OV_YY "+0.25)*y3(1)",
    "1.5*" OV_CXY "(1)/(" OV_YY "+0.25)",                  /* exb */
    "1.5*" OV_CXY "(2)/(" OV_YY "+0.25)",
    "1.5*" OV_CXY "(3)/(" OV_YY "+0.25)",
    "1.5*x1(1)*sqrt(" OV_SQR "+0.25)",                       /* drag */
    "1.5*x2(1)*sqrt(" OV_SQR "+0.25)",
    "1.5*x3(1)*sqrt(" OV_SQR "+0.25)",
    "q(1).*x1(1)", "q(1).*x2(1)", "q(1).*x3(1)",             /* mom */
    "0.5*q(1)*" OV_SQR,                                      /* ke */
    "0.25*((U(14)+U(20))+(U(10)+U(24))-4*U(17))",            /* lap5 */
    "0.25*((U(14)+U(20))+(U(10)+U(24))+(U(6)+U(28))-6*U(17))" /* lap7 */
};

/* complex-only overrides: the kernels whose scalar is complex, and
 * 1norm (abs() above is real-only; the kernel is sqrt(q*q)).  The
 * dot-family slots are NOT overridden - see the conjugation caveat
 * in the header comment before enabling c/z printing. */
static const char *ov_expr_c[50] = {
    [0]  = "(1.5+0.5i)*xv(1)+yv(1)",
    [1]  = "(q(1).*t(1))+(0.25+0.1i)",
    [7]  = "sqrt(q(1).*q(1))",
    [11] = "(1.5+0.5i)*" OV_CXY "(1)",
    [12] = "(1.5+0.5i)*" OV_CXY "(2)",
    [13] = "(1.5+0.5i)*" OV_CXY "(3)",
};

#undef OV_VX
#undef OV_VY
#undef OV_VW
#undef OV_DOT
#undef OV_SQR
#undef OV_XW
#undef OV_YY
#undef OV_CXY

static void ov_val(char *buf, size_t sz, cr v, int iscx)
{
    if (!iscx || v.im == 0)
        snprintf(buf, sz, "%.17g", (double)v.re);
    else
        snprintf(buf, sz, "(%.17g%+.17gi)", (double)v.re, (double)v.im);
}

/* formulas are written over tokens (below); ov_expand() replaces them
 * with printed-number literals at emit time: lowercase tokens match
 * the "x1(1)" form, uppercase ones name compound pieces. */
static const char *ov_toks[17] = {
    "x1(1)", "x2(1)", "x3(1)", "y1(1)", "y2(1)", "y3(1)",
    "w1(1)", "w2(1)", "w3(1)", "q(1)", "t(1)", "xv(1)", "yv(1)",
    "VX", "VY", "VW", "CXY"
};

static int ov_expand(char *buf, size_t sz, const char *repl[17])
{
    char tmp[2048];
    for (int pass = 0; pass < 8; pass++) {
        int n = 0, hit = 0;
        size_t i = 0;
        while (buf[i] && n < (int)sizeof tmp - 64) {
            int id = (i == 0) || !(isalnum((unsigned char)buf[i - 1]) ||
                                   buf[i - 1] == '_');
            int done = 0;
            if (id)
                for (int k = 0; k < 17 && !done; k++) {
                    size_t L = strlen(ov_toks[k]);
                    if (!strncmp(buf + i, ov_toks[k], L)) {
                        char nx = buf[i + L];
                        if (!(isalnum((unsigned char)nx) || nx == '_')) {
                            n += snprintf(tmp + n, sizeof tmp - n, "%s",
                                          repl[k]);
                            i += L;
                            hit = done = 1;
                        }
                    }
                }
            if (!done)
                tmp[n++] = buf[i++];
        }
        tmp[n] = 0;
        memcpy(buf, tmp, n + 1);
        if (!hit)
            return n < (int)sz ? 0 : -1;
    }
    return -1;
}

/*
 * in[]: 13 byte-pointers to the input arrays (order ov_in_names),
 * element 0 = first element.  u: padded lap buffer with element 0 at
 * index 16, or NULL to skip the lap rows.  nslot: 48 or 50.
 * d precision only: other precisions print nothing (see header).
 */
static void octave_dump(char pch, int esz, int iscx, int nslot,
                        const void *const *g, const void *const *in,
                        const void *u)
{
    static char fb[2048];
    char vb[64], nm[48], sv[64], f[13][64];
    char vx[192], vy[192], vw[192], cxy[448];
    const char *repl[17];
    cr v;
    char pstr[2] = { pch, 0 };

    if (pch != 'd')
        return;
    printf("\n%% ==== Octave/MATLAB check (element 1; save = kernel,"
           " formula = builtins) ====\n");
    for (int k = 0; k < 13; k++) {
        xval(esz, iscx, (const unsigned char *)in[k], &v);
        ov_val(f[k], sizeof f[k], v, iscx);
    }
    snprintf(vx, sizeof vx, "[%s,%s,%s]", f[0], f[1], f[2]);
    snprintf(vy, sizeof vy, "[%s,%s,%s]", f[3], f[4], f[5]);
    snprintf(vw, sizeof vw, "[%s,%s,%s]", f[6], f[7], f[8]);
    snprintf(cxy, sizeof cxy, "cross(%s,%s)", vx, vy);
    for (int k = 0; k < 13; k++)
        repl[k] = f[k];
    repl[13] = vx; repl[14] = vy; repl[15] = vw; repl[16] = cxy;
    if (nslot > 48 && u) {
        printf("U=[");
        for (int i = -16; i < 8 + 16; i++) {
            xval(esz, iscx, (const unsigned char *)u + (size_t)(i + 16) * esz, &v);
            ov_val(vb, sizeof vb, v, iscx);
            printf("%s%s", i == -16 ? "" : ",", vb);
        }
        printf("]; %% U(17) = element 1, pad 16\n");
    }
    for (int k = 0; k < nslot; k++) {
        const char *ex = (iscx && ov_expr_c[k]) ? ov_expr_c[k] : ov_expr_r[k];
        snprintf(nm, sizeof nm, ov_names[k], pstr);
        char *par = strchr(nm, '(');
        if (par)
            snprintf(sv, sizeof sv, "%.*s_save%s", (int)(par - nm), nm, par);
        else
            snprintf(sv, sizeof sv, "%s_save", nm);
        xval(esz, iscx, (const unsigned char *)g[k], &v);
        ov_val(vb, sizeof vb, v, iscx);
        snprintf(fb, sizeof fb, "%s", ex);
        if (ov_expand(fb, sizeof fb, repl))
            printf("%% %s: formula expansion overflowed\n", nm);
        printf("%s = %s\n", sv, vb);   /* no ; : must echo  */
        printf("%s = %s\n", nm, fb);
    }
    printf("%% =====================================================================\n");
}

#endif /* OPS_OCTAVE_H */
