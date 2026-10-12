/*
 * ops_octave.h - copy-paste verification block for the physics kernels.
 *
 * Prints, for the FIRST element of every output, an Octave/MATLAB line
 * "name = formula;" that recomputes it from the printed inputs, plus a
 * "% => value" comment with what the kernel actually produced.  All
 * products are bilinear (no conjugation), which is why one formula set
 * serves real and complex: paste into Octave, compare by eye.
 * Used by cblas_test.c and v3blas_test.c only - never by benchmarks.
 *
 * Requires the xval()/cr machinery of the including test.
 */
#ifndef OPS_OCTAVE_H
#define OPS_OCTAVE_H

static const char *ov_in_names[13] = {
    "x1", "x2", "x3", "y1", "y2", "y3", "w1", "w2", "w3", "q", "t",
    "xv", "yv"
};

/* slot order matches the g[] arrays of the tests:
 *  0 axpy, 1 1xypa, 2 3dot, 3-5 3had, 6 3sqr, 7 1norm, 8-10 3cross,
 * 11-13 3crossscal, 14 3crossdot, 15 3crosssqr, 16-18 crossxy_crossxz.u,
 * 19-21 crossxy_crossxz.v, 22-24 crossxy_dotxz.u, 25 crossxy_dotxz.r,
 * 26 dotxy_dotxz.r, 27 dotxy_dotxz.q, 28-30 crosscross, 31 norm_unit.r,
 * 32-34 norm_unit.u, 35-37 refl, 38-40 exb, 41-43 drag, 44-46 mom,
 * 47 ke, 48 lap5, 49 lap7.
 * %s = precision prefix (s/d/c/z).  cx form used for complex when it
 * differs (a = 1.5+0.5i for axpy/crossscal, 0.25+0.1i for 1xypa). */
#define OV_DOT "(x1(1).*y1(1)+x2(1).*y2(1)+x3(1).*y3(1))"
#define OV_SQR "(x1(1).*x1(1)+x2(1).*x2(1)+x3(1).*x3(1))"
#define OV_XW  "(x1(1).*w1(1)+x2(1).*w2(1)+x3(1).*w3(1))"
#define OV_YY  "(y1(1).*y1(1)+y2(1).*y2(1)+y3(1).*y3(1))"
#define OV_C1  "(x2(1).*y3(1)-x3(1).*y2(1))"
#define OV_C2  "(x3(1).*y1(1)-x1(1).*y3(1))"
#define OV_C3  "(x1(1).*y2(1)-x2(1).*y1(1))"
#define OV_W1  "(x2(1).*w3(1)-x3(1).*w2(1))"
#define OV_W2  "(x3(1).*w1(1)-x1(1).*w3(1))"
#define OV_W3  "(x1(1).*w2(1)-x2(1).*w1(1))"

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
    "sqrt(q(1).*q(1))",                                      /* 1norm */
    OV_C1, OV_C2, OV_C3,                                     /* 3cross */
    "1.5*" OV_C1, "1.5*" OV_C2, "1.5*" OV_C3,                /* crossscal */
    OV_C1 "*w1(1)+" OV_C2 "*w2(1)+" OV_C3 "*w3(1)",          /* crossdot */
    OV_C1 "*" OV_C1 "+" OV_C2 "*" OV_C2 "+" OV_C3 "*" OV_C3, /* crosssqr */
    OV_C1, OV_C2, OV_C3,                                     /* cxy.u */
    OV_W1, OV_W2, OV_W3,                                     /* cxy.v */
    OV_C1, OV_C2, OV_C3, OV_XW,                              /* cxdot */
    OV_DOT, OV_XW,                                           /* dxdx r,q */
    "(" OV_C2 ")*w3(1)-(" OV_C3 ")*w2(1)",                   /* crosscross */
    "(" OV_C3 ")*w1(1)-(" OV_C1 ")*w3(1)",
    "(" OV_C1 ")*w2(1)-(" OV_C2 ")*w1(1)",
    "sqrt(" OV_SQR "+0.25)",                                 /* norm_unit r */
    "x1(1)/sqrt(" OV_SQR "+0.25)",
    "x2(1)/sqrt(" OV_SQR "+0.25)",
    "x3(1)/sqrt(" OV_SQR "+0.25)",
    "x1(1)-1.5*" OV_DOT "/(" OV_YY "+0.25)*y1(1)",           /* refl */
    "x2(1)-1.5*" OV_DOT "/(" OV_YY "+0.25)*y2(1)",
    "x3(1)-1.5*" OV_DOT "/(" OV_YY "+0.25)*y3(1)",
    "1.5*" OV_C1 "/(" OV_YY "+0.25)",                        /* exb */
    "1.5*" OV_C2 "/(" OV_YY "+0.25)",
    "1.5*" OV_C3 "/(" OV_YY "+0.25)",
    "1.5*x1(1)*sqrt(" OV_SQR "+0.25)",                       /* drag */
    "1.5*x2(1)*sqrt(" OV_SQR "+0.25)",
    "1.5*x3(1)*sqrt(" OV_SQR "+0.25)",
    "q(1).*x1(1)", "q(1).*x2(1)", "q(1).*x3(1)",             /* mom */
    "0.5*q(1)*" OV_SQR,                                      /* ke */
    "0.25*((U(14)+U(20))+(U(10)+U(24))-4*U(17))",            /* lap5 */
    "0.25*((U(14)+U(20))+(U(10)+U(24))+(U(6)+U(28))-6*U(17))" /* lap7 */
};

/* complex-only overrides: the three kernels whose scalar is complex */
static const char *ov_expr_c[50] = {
    [0]  = "(1.5+0.5i)*xv(1)+yv(1)",
    [1]  = "(q(1).*t(1))+(0.25+0.1i)",
    [11] = "(1.5+0.5i)*" OV_C1,
    [12] = "(1.5+0.5i)*" OV_C2,
    [13] = "(1.5+0.5i)*" OV_C3,
};

#undef OV_DOT
#undef OV_SQR
#undef OV_XW
#undef OV_YY
#undef OV_C1
#undef OV_C2
#undef OV_C3
#undef OV_W1
#undef OV_W2
#undef OV_W3

static void ov_val(char *buf, size_t sz, cr v, int iscx)
{
    if (!iscx || v.im == 0)
        snprintf(buf, sz, "%.17g", (double)v.re);
    else
        snprintf(buf, sz, "(%.17g%+.17gi)", (double)v.re, (double)v.im);
}

/*
 * in[]: 13 byte-pointers to the input arrays (order ov_in_names),
 * element 0 = first element.  u: padded lap buffer with element 0 at
 * index 16, or NULL to skip the lap rows.  nslot: 48 or 50.
 */
static void octave_dump(char pch, int esz, int iscx, int nslot,
                        const void *const *g, const void *const *in,
                        const void *u)
{
    char vb[64], nm[48];
    cr v;
    char pstr[2] = { pch, 0 };
    printf("\n%% ==== Octave/MATLAB check (element 1; .* keeps products bilinear) ====\n");
    for (int k = 0; k < 13; k++) {
        xval(esz, iscx, (const unsigned char *)in[k], &v);
        ov_val(vb, sizeof vb, v, iscx);
        printf("%s(1)=%s;\n", ov_in_names[k], vb);
    }
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
        xval(esz, iscx, (const unsigned char *)g[k], &v);
        ov_val(vb, sizeof vb, v, iscx);
        printf("%s = %s;   %% => %s\n", nm, ex, vb);
    }
    printf("%% =====================================================================\n");
}

#endif /* OPS_OCTAVE_H */
