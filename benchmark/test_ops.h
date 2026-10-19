/*
 * test_ops.h - shared verification engine for the physics extension
 * tests (cblas_test.c, v3blas_test.c): long double reference values for
 * all element-wise ops, seeded input data, relative-error check.
 * Requires <stdio.h>, <string.h>, <math.h>, the includer's globals
 * `prec` (char *) and VAL; N is defined by the includer.
 * Never used by benchmarks: correctness lives here, timing lives there.
 */
#ifndef TEST_OPS_H
#define TEST_OPS_H

/* complex PODs when the includer has no v3blas.h (cblas-only test) */
#define N 8

#ifndef V3BLAS_H
typedef struct {
  float re, im;
} v3blas_floatcomplex;
typedef struct {
  double re, im;
} v3blas_doublecomplex;
#endif

typedef long double ld;

#define VAL(S, i) ((ld)((S) + (i)) * 0.1L * (((i) % 3) == 0 ? -1.0L : 1.0L))

static int fails = 0;
static const char *prec = "?";

static void chk(const char *op, ld e, ld eps) {
  if (e > eps) {
    printf("%% %s %-24s max_rel_err = %.3e  FAIL\n", prec, op, (double)e);
    fails++;
  } else {
    printf("%% %s %-24s max_rel_err = %.3e  ok\n", prec, op, (double)e);
  }
}

static ld rabs(ld a) { return a < 0 ? -a : a; }
static ld m1(ld a) {
  a = rabs(a);
  return a < 1 ? 1 : a;
}
static ld mx(ld a, ld b) { return a > b ? a : b; }

/* long double "complex real" arithmetic for the reference */
typedef struct {
  ld re, im;
} cr;

static cr cmul(cr a, cr b) {
  cr c;
  c.re = a.re * b.re - a.im * b.im;
  c.im = a.re * b.im + a.im * b.re;
  return c;
}
static cr cadd(cr a, cr b) {
  cr c = {a.re + b.re, a.im + b.im};
  return c;
}
static cr csub(cr a, cr b) {
  cr c = {a.re - b.re, a.im - b.im};
  return c;
}
static cr cdiv(cr a, cr b) {
  ld d = b.re * b.re + b.im * b.im;
  cr c;
  c.re = (a.re * b.re + a.im * b.im) / d;
  c.im = (a.im * b.re - a.re * b.im) / d;
  return c;
}

static void ccross3(const cr x[3], const cr y[3], cr w[3]) {
  cr p = cmul(x[1], y[2]);
  cr q = cmul(x[2], y[1]);
  w[0].re = p.re - q.re;
  w[0].im = p.im - q.im;
  p = cmul(x[2], y[0]);
  q = cmul(x[0], y[2]);
  w[1].re = p.re - q.re;
  w[1].im = p.im - q.im;
  p = cmul(x[0], y[1]);
  q = cmul(x[1], y[0]);
  w[2].re = p.re - q.re;
  w[2].im = p.im - q.im;
}

static cr cdot3c(const cr x[3], const cr w[3]) {
  cr s = {0, 0}, p;
  for (int c = 0; c < 3; c++) {
    p = cmul(x[c], w[c]);
    s = cadd(s, p);
  }
  return s;
}

static cr cnorm1(cr q) {
  ld p0 = q.re * q.re - q.im * q.im;
  ld p1 = 2.0L * q.re * q.im;
  ld m = sqrtl(p0 * p0 + p1 * p1);
  cr r;
  r.re = sqrtl((m + p0) / 2.0L);
  r.im = (p1 >= 0) ? sqrtl((m - p0) / 2.0L) : -sqrtl((m - p0) / 2.0L);
  return r;
}

static cr csqrt_cr(cr z) {
  ld m = sqrtl(z.re * z.re + z.im * z.im);
  cr r;
  r.re = sqrtl((m + z.re) / 2.0L);
  r.im = (z.im >= 0) ? sqrtl((m - z.re) / 2.0L) : -sqrtl((m - z.re) / 2.0L);
  return r;
}

/* reference values for element i; seeds: x=(b,b+1,b+2), y=(b+3..b+5),
 * w=(b+6..b+8), q=b+9, t=b+10, yv=b+11, xv=b+12 */
typedef struct {
  cr ax, xy, dot, sqr, norm, crd, crsq, xw;
  cr had[3], crw[3], crs[3], u[3], v[3];
  cr cc[3], nur, nu[3], rf[3], ex[3], dr[3], mm[3], kk;
} refset;

static void refset_fill(int i, int b, int iscx, refset *R) {
  cr x[3], y[3], w[3], q, t, xv, yv, s, a;
  ld im = iscx ? 0.7L : 0.0L;
  for (int c = 0; c < 3; c++) {
    ld v = VAL(b + c, i);
    x[c] = (cr){v, v * im};
    v = VAL(b + 3 + c, i);
    y[c] = (cr){v, v * im};
    v = VAL(b + 6 + c, i);
    w[c] = (cr){v, v * im};
  }
  q = (cr){VAL(b + 9, i), VAL(b + 9, i) * im};
  t = (cr){VAL(b + 10, i), VAL(b + 10, i) * im};
  xv = (cr){VAL(b + 12, i), VAL(b + 12, i) * im};
  yv = (cr){VAL(b + 11, i), VAL(b + 11, i) * im};
  a = (cr){1.5L, iscx ? 0.5L : 0.0L};
  R->ax = cadd(cmul(a, xv), yv);
  R->xy = cadd(cmul(q, t), (cr){0.25L, iscx ? 0.1L : 0.0L});
  R->dot = cdot3c(x, y);
  for (int c = 0; c < 3; c++) R->had[c] = cmul(x[c], y[c]);
  s = (cr){0, 0};
  for (int c = 0; c < 3; c++) s = cadd(s, cmul(x[c], x[c]));
  R->sqr = s;
  R->norm = cnorm1(q);
  ccross3(x, y, R->crw);
  for (int c = 0; c < 3; c++) R->crs[c] = cmul(a, R->crw[c]);
  R->crd = cdot3c(R->crw, w);
  R->crsq = cdot3c(R->crw, R->crw);
  ccross3(x, y, R->u);
  ccross3(x, w, R->v);
  R->xw = cdot3c(x, w);
  /* new CFD ops: real a = 1.5, eps = 0.25 in every precision */
  {
    cr tmp[3], epsc = {0.25L, 0.0L}, asc = {1.5L, 0.0L};
    cr half = {0.5L, 0.0L}, sxx, dyy, ff;
    ccross3(x, y, tmp);
    ccross3(tmp, w, R->cc);
    sxx = cdot3c(x, x);
    R->nur = csqrt_cr(cadd(sxx, epsc)); /* sqrt(x.x + eps) */
    for (int c = 0; c < 3; c++) R->nu[c] = cdiv(x[c], R->nur);
    dyy = cdot3c(y, y);
    ff = cdiv(cmul(asc, cdot3c(x, y)), cadd(dyy, epsc));
    for (int c = 0; c < 3; c++) R->rf[c] = csub(x[c], cmul(ff, y[c]));
    ff = cdiv(asc, cadd(dyy, epsc));
    for (int c = 0; c < 3; c++) R->ex[c] = cmul(ff, R->crw[c]);
    for (int c = 0; c < 3; c++) R->dr[c] = cmul(cmul(asc, x[c]), R->nur);
    for (int c = 0; c < 3; c++) R->mm[c] = cmul(q, x[c]);
    R->kk = cmul(cmul(half, q), sxx);
  }
}

/* extract one result element (re, im) from a kernel output array */
static void xval(int esz, int iscx, const unsigned char *p, cr *v) {
  if (esz == 4) {
    float f;
    memcpy(&f, p, 4);
    v->re = f;
    v->im = 0;
  } else if (esz == 8 && !iscx) {
    double d;
    memcpy(&d, p, 8);
    v->re = d;
    v->im = 0;
  } else if (esz == 8) {
    float a, b;
    memcpy(&a, p, 4);
    memcpy(&b, p + 4, 4);
    v->re = a;
    v->im = b;
  } else {
    double a, b;
    memcpy(&a, p, 8);
    memcpy(&b, p + 8, 8);
    v->re = a;
    v->im = b;
  }
}

static ld relerr(cr g, cr r) { return mx(rabs(g.re - r.re) / m1(r.re), rabs(g.im - r.im) / m1(r.im)); }

static cr field(const refset *r, int k) {
  switch (k) {
  case 0: return r->ax;
  case 1: return r->xy;
  case 2: return r->dot;
  case 3: return r->had[0];
  case 4: return r->had[1];
  case 5: return r->had[2];
  case 6: return r->sqr;
  case 7: return r->norm;
  case 8: return r->crw[0];
  case 9: return r->crw[1];
  case 10: return r->crw[2];
  case 11: return r->crs[0];
  case 12: return r->crs[1];
  case 13: return r->crs[2];
  case 14: return r->crd;
  case 15: return r->crsq;
  case 16: return r->u[0];
  case 17: return r->u[1];
  case 18: return r->u[2];
  case 19: return r->v[0];
  case 20: return r->v[1];
  case 21: return r->v[2];
  case 22: return r->u[0];
  case 23: return r->u[1];
  case 24: return r->u[2];
  case 25: return r->xw;
  case 26: return r->dot;
  case 28: return r->cc[0];
  case 29: return r->cc[1];
  case 30: return r->cc[2];
  case 31: return r->nur;
  case 32: return r->nu[0];
  case 33: return r->nu[1];
  case 34: return r->nu[2];
  case 35: return r->rf[0];
  case 36: return r->rf[1];
  case 37: return r->rf[2];
  case 38: return r->ex[0];
  case 39: return r->ex[1];
  case 40: return r->ex[2];
  case 41: return r->dr[0];
  case 42: return r->dr[1];
  case 43: return r->dr[2];
  case 44: return r->mm[0];
  case 45: return r->mm[1];
  case 46: return r->mm[2];
  case 47: return r->kk;
  default: return r->xw;
  }
}

static const char *opnames[48] = {"%saxpy",
                                  "%s1xypa",
                                  "%s3dot",
                                  "%s3had.x",
                                  "%s3had.y",
                                  "%s3had.z",
                                  "%s3sqr",
                                  "%s1norm",
                                  "%s3cross.x",
                                  "%s3cross.y",
                                  "%s3cross.z",
                                  "%s3crossscal.x",
                                  "%s3crossscal.y",
                                  "%s3crossscal.z",
                                  "%s3crossdot",
                                  "%s3crosssqr",
                                  "%s3crossxy_crossxz.u.x",
                                  "%s3crossxy_crossxz.u.y",
                                  "%s3crossxy_crossxz.u.z",
                                  "%s3crossxy_crossxz.v.x",
                                  "%s3crossxy_crossxz.v.y",
                                  "%s3crossxy_crossxz.v.z",
                                  "%s3crossxy_dotxz.u.x",
                                  "%s3crossxy_dotxz.u.y",
                                  "%s3crossxy_dotxz.u.z",
                                  "%s3crossxy_dotxz.r",
                                  "%s3dotxy_dotxz.r",
                                  "%s3dotxy_dotxz.q",
                                  "%s3crosscross.x",
                                  "%s3crosscross.y",
                                  "%s3crosscross.z",
                                  "%s3norm_unit.r",
                                  "%s3norm_unit.u.x",
                                  "%s3norm_unit.u.y",
                                  "%s3norm_unit.u.z",
                                  "%s3refl.x",
                                  "%s3refl.y",
                                  "%s3refl.z",
                                  "%s3exb.x",
                                  "%s3exb.y",
                                  "%s3exb.z",
                                  "%s3drag.x",
                                  "%s3drag.y",
                                  "%s3drag.z",
                                  "%s3mom_ke.x",
                                  "%s3mom_ke.y",
                                  "%s3mom_ke.z",
                                  "%s3mom_ke.k"};

static void cmp_all(int esz, int iscx, const void *const g[48], const refset *R, int n, ld eps) {
  for (int k = 0; k < 48; k++) {
    ld e = 0;
    for (int i = 0; i < n; i++) {
      cr gv, rv = field(&R[i], k);
      xval(esz, iscx, (const unsigned char *)g[k] + (size_t)i * esz, &gv);
      if (relerr(gv, rv) > e) e = relerr(gv, rv);
    }
    char nm[48];
    snprintf(nm, sizeof nm, opnames[k], prec);
    chk(nm, e, eps);
  }
}

/* data fills: components of a 3-vector get seeds S, S+1, S+2 */
static void f1_s(float *v, int S) {
  for (int i = 0; i < N; i++) v[i] = (float)VAL(S, i);
}
static void f1_d(double *v, int S) {
  for (int i = 0; i < N; i++) v[i] = (double)VAL(S, i);
}
static void f1_c(v3blas_floatcomplex *v, int S) {
  for (int i = 0; i < N; i++) {
    v[i].re = (float)VAL(S, i);
    v[i].im = (float)(VAL(S, i) * 0.7L);
  }
}
static void f1_z(v3blas_doublecomplex *v, int S) {
  for (int i = 0; i < N; i++) {
    v[i].re = VAL(S, i);
    v[i].im = VAL(S, i) * 0.7L;
  }
}

static void f3_s(float *a0, float *a1, float *a2, int S) {
  for (int i = 0; i < N; i++) {
    a0[i] = (float)VAL(S, i);
    a1[i] = (float)VAL(S + 1, i);
    a2[i] = (float)VAL(S + 2, i);
  }
}
static void f3_d(double *a0, double *a1, double *a2, int S) {
  for (int i = 0; i < N; i++) {
    a0[i] = (double)VAL(S, i);
    a1[i] = (double)VAL(S + 1, i);
    a2[i] = (double)VAL(S + 2, i);
  }
}
static void f3_c(v3blas_floatcomplex *a0, v3blas_floatcomplex *a1, v3blas_floatcomplex *a2, int S) {
  for (int i = 0; i < N; i++) {
    a0[i].re = (float)VAL(S, i);
    a0[i].im = (float)(VAL(S, i) * 0.7L);
    a1[i].re = (float)VAL(S + 1, i);
    a1[i].im = (float)(VAL(S + 1, i) * 0.7L);
    a2[i].re = (float)VAL(S + 2, i);
    a2[i].im = (float)(VAL(S + 2, i) * 0.7L);
  }
}
static void f3_z(v3blas_doublecomplex *a0, v3blas_doublecomplex *a1, v3blas_doublecomplex *a2, int S) {
  for (int i = 0; i < N; i++) {
    a0[i].re = VAL(S, i);
    a0[i].im = VAL(S, i) * 0.7L;
    a1[i].re = VAL(S + 1, i);
    a1[i].im = VAL(S + 1, i) * 0.7L;
    a2[i].re = VAL(S + 2, i);
    a2[i].im = VAL(S + 2, i) * 0.7L;
  }
}

#endif /* TEST_OPS_H */
