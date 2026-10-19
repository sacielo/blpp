/*
 * ops_octave.h - Octave/MATLAB check block for the physics kernels.
 *
 * Printed ALWAYS (no flag), at d precision only: the other precisions
 * return early.  The point is that the test's whole stdout is then a
 * valid .m file:  ./cblas_test d > check.m && octave check.m
 * Every human-readable message in the tests is a % comment for this
 * reason; nothing else may be printed.
 *
 * One PAIR of lines per output, vectors intact - Octave is not C:
 *     d3cross_save = [-0.03,0.06,-0.03]      <- what the kernel gave
 *     d3cross = cross([..],[..])             <- builtin redoes it
 * NO trailing semicolon: Octave/MATLAB must echo both lines so the
 * printed pair can be compared by eye (a silent run proves nothing).
 * Inputs are inlined as printed numbers, no variable preamble; the
 * only shared line is U=[...] for the lap5/lap7 stencils (40 numbers,
 * cited 9 times, kept silent with ;).
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

/* slot order matches the g[] arrays of the tests (axpy=0, 1xypa=1,
 * 3dot=2, 3had=3..5, 3sqr=6, 1norm=7, 3cross=8..10, crossscal=11..13,
 * crossdot=14, crosssqr=15, crossxy_crossxz.u=16..18 .v=19..21,
 * crossxy_dotxz.u=22..24 .r=25, dotxy_dotxz.r=26 .q=27,
 * crosscross=28..30, norm_unit.r=31 .u=32..34, refl=35..37, exb=38..40,
 * drag=41..43, mom=44..46, ke=47, lap5=48, lap7=49).  %s = precision
 * prefix.  Scalars are baked in (a = 1.5, eps = 0.25; 0.25+0.1i for
 * 1xypa, 1.5+0.5i for axpy/crossscal).  Outputs are scalars or whole
 * 3-vectors (ov_out_ncomp); names carry no indices. */
#define OV_VX "[x1(1),x2(1),x3(1)]"
#define OV_VY "[y1(1),y2(1),y3(1)]"
#define OV_VW "[w1(1),w2(1),w3(1)]"
#define OV_DOT "dot(" OV_VX "," OV_VY ")"
#define OV_SQR "dot(" OV_VX "," OV_VX ")"
#define OV_XW "dot(" OV_VX "," OV_VW ")"
#define OV_YY "dot(" OV_VY "," OV_VY ")"
#define OV_CXY "cross(" OV_VX "," OV_VY ")"

static const char *ov_out_names[26] = {"%saxpy",        "%s1xypa",      "%s3dot",         "%s3had",  "%s3sqr", "%s1norm", "%s3cross", "%s3crossscal", "%s3crossdot", "%s3crosssqr", "%scrossxy_crossxz_u", "%scrossxy_crossxz_v", "%scrossxy_dotxz_u", "%scrossxy_dotxz_r", "%sdotxy_dotxz_r", "%sdotxy_dotxz_q",
                                       "%s3crosscross", "%s3norm_unit", "%s3norm_unit_u", "%s3refl", "%s3exb", "%s3drag", "%s3mom",   "%s3ke",        "%slap5",      "%slap7"};

static const unsigned char ov_out_slot[26] = {0, 1, 2, 3, 6, 7, 8, 11, 14, 15, 16, 19, 22, 25, 26, 27, 28, 31, 32, 35, 38, 41, 44, 47, 48, 49};

static const unsigned char ov_out_ncomp[26] = {1, 1, 1, 3, 1, 1, 3, 3, 1, 1, 3, 3, 3, 1, 1, 1, 3, 1, 3, 3, 3, 3, 3, 1, 1, 1};

static const char *ov_out_expr_r[26] = {
  "1.5*xv(1)+yv(1)",            /* axpy */
  "q(1).*t(1)+0.25",            /* 1xypa */
  OV_DOT,                       /* 3dot */
  OV_VX ".*" OV_VY,             /* 3had */
  OV_SQR,                       /* 3sqr */
  "abs(q(1))",                  /* 1norm */
  OV_CXY,                       /* 3cross */
  "1.5*" OV_CXY,                /* crossscal */
  "dot(" OV_CXY "," OV_VW ")",  /* crossdot */
  "dot(" OV_CXY "," OV_CXY ")", /* crosssqr */
  OV_CXY,                       /* cxcxz.u */
  "cross(" OV_VX "," OV_VW ")", /* cxcxz.v */
  OV_CXY,                       /* cxdot.u */
  "dot(" OV_CXY "," OV_VW ")",  /* cxdot.r */
  OV_DOT,
  OV_XW,                                                    /* dxdx r,q */
  "cross(" OV_CXY "," OV_VW ")",                            /* crosscross */
  "sqrt(" OV_SQR "+0.25)",                                  /* norm_unit r */
  OV_VX "./sqrt(" OV_SQR "+0.25)",                          /* norm_unit u */
  OV_VX "-1.5*" OV_DOT "/(" OV_YY "+0.25)*" OV_VY,          /* refl */
  "1.5*" OV_CXY "/(" OV_YY "+0.25)",                        /* exb */
  "1.5*" OV_VX "*sqrt(" OV_SQR "+0.25)",                    /* drag */
  "q(1).*" OV_VX,                                           /* mom */
  "0.5*q(1)*" OV_SQR,                                       /* ke */
  "0.25*((U(14)+U(20))+(U(10)+U(24))-4*U(17))",             /* lap5 */
  "0.25*((U(14)+U(20))+(U(10)+U(24))+(U(6)+U(28))-6*U(17))" /* lap7 */
};

/* complex-only overrides: the kernels whose scalar is complex, and
 * 1norm (abs() above is real-only; the kernel is sqrt(q*q)).  The
 * dot-family entries are NOT overridden - see the conjugation caveat
 * in the header comment before enabling c/z printing. */
static const char *ov_out_expr_c[26] = {
  [0] = "(1.5+0.5i)*xv(1)+yv(1)",
  [1] = "(q(1).*t(1))+(0.25+0.1i)",
  [5] = "sqrt(q(1).*q(1))",
  [7] = "(1.5+0.5i)*" OV_CXY,
};

#undef OV_VX
#undef OV_VY
#undef OV_VW
#undef OV_DOT
#undef OV_SQR
#undef OV_XW
#undef OV_YY
#undef OV_CXY

/* formulas are written over tokens (below); ov_expand() replaces them
 * with printed-number literals at emit time: lowercase tokens match
 * the "x1(1)" form, uppercase ones name compound pieces. */
static const char *ov_toks[17] = {"x1(1)", "x2(1)", "x3(1)", "y1(1)", "y2(1)", "y3(1)", "w1(1)", "w2(1)", "w3(1)", "q(1)", "t(1)", "xv(1)", "yv(1)", "VX", "VY", "VW", "CXY"};

static int ov_expand(char *buf, size_t sz, const char *repl[17]) {
  char tmp[2048];
  for (int pass = 0; pass < 8; pass++) {
    int n = 0, hit = 0;
    size_t i = 0;
    while (buf[i] && n < (int)sizeof tmp - 64) {
      int id = (i == 0) || !(isalnum((unsigned char)buf[i - 1]) || buf[i - 1] == '_');
      int done = 0;
      if (id)
        for (int k = 0; k < 17 && !done; k++) {
          size_t L = strlen(ov_toks[k]);
          if (!strncmp(buf + i, ov_toks[k], L)) {
            char nx = buf[i + L];
            if (!(isalnum((unsigned char)nx) || nx == '_')) {
              n += snprintf(tmp + n, sizeof tmp - n, "%s", repl[k]);
              i += L;
              hit = done = 1;
            }
          }
        }
      if (!done) tmp[n++] = buf[i++];
    }
    tmp[n] = 0;
    memcpy(buf, tmp, n + 1);
    if (!hit) return n < (int)sz ? 0 : -1;
  }
  return -1;
}

static void ov_val(char *buf, size_t sz, cr v, int iscx) {
  if (!iscx || v.im == 0) snprintf(buf, sz, "%.17g", (double)v.re);
  else
    snprintf(buf, sz, "(%.17g%+.17gi)", (double)v.re, (double)v.im);
}

/*
 * in[]: 13 byte-pointers to the input arrays (order ov_toks),
 * element 0 = first element.  g[]: kernel outputs, slot order as in
 * the comment above.  u: padded lap buffer with element 0 at index
 * 16, or NULL to skip the lap rows.  nslot: 48 or 50 (lap entries
 * beyond it are skipped).  d precision only: other precisions print
 * nothing (see header).
 */
static void octave_dump(char pch, int esz, int iscx, int nslot, const void *const *g, const void *const *in, const void *u) {
  static char fb[2048];
  char vb[64], nm[48], sv[64], f[13][64];
  char vx[192], vy[192], vw[192], cxy[448];
  const char *repl[17];
  cr v;
  char pstr[2] = {pch, 0};

  if (pch != 'd') return;
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
  for (int k = 0; k < 13; k++) repl[k] = f[k];
  repl[13] = vx;
  repl[14] = vy;
  repl[15] = vw;
  repl[16] = cxy;
  if (nslot > 48 && u) {
    printf("U=[");
    for (int i = -16; i < 8 + 16; i++) {
      xval(esz, iscx, (const unsigned char *)u + (size_t)(i + 16) * esz, &v);
      ov_val(vb, sizeof vb, v, iscx);
      printf("%s%s", i == -16 ? "" : ",", vb);
    }
    printf("]; %% U(17) = element 1, pad 16\n");
  }
  for (int k = 0; k < 26; k++) {
    int slot = ov_out_slot[k], nc = ov_out_ncomp[k];
    const char *ex = (iscx && ov_out_expr_c[k]) ? ov_out_expr_c[k] : ov_out_expr_r[k];
    if (slot + nc > nslot) continue;
    snprintf(nm, sizeof nm, ov_out_names[k], pstr);
    snprintf(sv, sizeof sv, "%s_save", nm);
    if (nc == 1) {
      xval(esz, iscx, (const unsigned char *)g[slot], &v);
      ov_val(vb, sizeof vb, v, iscx);
      snprintf(fb, sizeof fb, "%s", vb);
    } else {
      char c[3][64];
      for (int j = 0; j < 3; j++) {
        xval(esz, iscx, (const unsigned char *)g[slot + j], &v);
        ov_val(c[j], sizeof c[j], v, iscx);
      }
      snprintf(fb, sizeof fb, "[%s,%s,%s]", c[0], c[1], c[2]);
    }
    printf("%s = %s\n", sv, fb); /* no ; : must echo */
    snprintf(fb, sizeof fb, "%s", ex);
    if (ov_expand(fb, sizeof fb, repl)) printf("%% %s: formula expansion overflowed\n", nm);
    else
      printf("%s = %s\n", nm, fb);
  }
  printf("%% =====================================================================\n");
}

#endif /* OPS_OCTAVE_H */
