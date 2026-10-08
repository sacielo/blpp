! v3blas_f90_test.f90 - correctness test for the v3blas Fortran front-end.
! All 13 operations are called through their generic names at all four
! precisions (the compiler picks s/d/c/z from the argument types) and the
! outputs are compared against the defining equation recomputed with
! Fortran array expressions.  Generated file: edit the templates, not this.
! Build: gfortran -Wall ../openblas/v3blas.f90 v3blas_f90_test.f90 \
!        -L../pkgs/openblas/lib -lopenblas -lm -o v3blas_f90_test

module v3blas_f90_cases
  use v3blas
  use, intrinsic :: iso_c_binding, only: c_int
  implicit none
  integer, public :: fails = 0
  private :: rfill_s, rfill_d, setup9_s, setup9_d
  private :: cfill_c, cfill_z, setup9c_c, setup9c_z
  public :: chk, test_real_s, test_real_d, test_cplx_c, test_cplx_z
contains

  subroutine chk(name, err, tol)
    character(len=*), intent(in) :: name
    real, intent(in) :: err, tol
    if (err > tol) then
       fails = fails + 1
       write(*, '(a, a, es10.1, a)') trim(name), '  FAIL err=', err
    else
       write(*, '(a, a, es10.1, a)') trim(name), '  ok   err=', err
    end if
  end subroutine chk


! Included from v3blas_f90_test.f90 with real(4) (type), TAG (name suffix),
! PK (printed precision tag) and 1.0E-5 (tolerance) defined.
! Every generic op is called through its generic name and checked against
! the defining equation recomputed with Fortran array expressions.

  subroutine rfill_s(c, v)
    real(4), intent(in) :: c
    real(4) :: v(:)
    integer :: i
    do i = 1, size(v)
       v(i) = real(i - 1, kind(v))/real(16.0, kind(v)) + c
    end do
  end subroutine rfill_s

  subroutine setup9_s(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    real(4) :: x1(:), x2(:), x3(:), y1(:), y2(:), y3(:), w1(:), w2(:), w3(:)
    real(4), parameter :: o(9) = [ 0.0, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.45, 0.5 ]
    call rfill_s(o(1), x1); call rfill_s(o(2), x2)
    call rfill_s(o(3), x3); call rfill_s(o(4), y1)
    call rfill_s(o(5), y2); call rfill_s(o(6), y3)
    call rfill_s(o(7), w1); call rfill_s(o(8), w2)
    call rfill_s(o(9), w3)
  end subroutine setup9_s

  subroutine test_real_s(n)
    integer(c_int), intent(in) :: n
    real(4) :: x1(n), x2(n), x3(n), y1(n), y2(n), y3(n)
    real(4) :: w1(n), w2(n), w3(n), u1(n), u2(n), u3(n), v1(n), v2(n), v3(n)
    real(4) :: r(n), q(n), t(n), cx(n), cy(n), cz(n), al, a2
    real(kind(u1)) :: e1, e2
    al = real(1.5, kind(al)); a2 = real(0.25, kind(a2))

    call setup9_s(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    t = y1
    call v1axpy(n, al, x1, y1)
    call chk('v1axpy  y=a*x+y ('//"s"//')', &
             real(maxval(abs(y1 - (al*x1 + t)))), 1.0E-5)

    call rfill_s(real(0.1, kind(q)), q)
    call rfill_s(real(-0.7, kind(t)), t)
    call v1xypa(n, a2, q, t, r)
    call chk('v1xypa  r=q*t+a ('//"s"//')', &
             real(maxval(abs(r - (q*t + a2)))), 1.0E-5)

    call setup9_s(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    call v3dot(n, x1, x2, x3, y1, y2, y3, r)
    call chk('v3dot   r=x.y ('//"s"//')', &
             real(maxval(abs(r - (x1*y1 + x2*y2 + x3*y3)))), 1.0E-5)

    call v3had(n, x1, x2, x3, y1, y2, y3, w1, w2, w3)
    call chk('v3had   w=x.*y ('//"s"//')', real(max(maxval(abs(w1 - x1*y1)), &
             max(maxval(abs(w2 - x2*y2)), maxval(abs(w3 - x3*y3))))), 1.0E-5)

    call v3sqr(n, x1, x2, x3, r)
    call chk('v3sqr   r=x.x ('//"s"//')', &
             real(maxval(abs(r - (x1*x1 + x2*x2 + x3*x3)))), 1.0E-5)

    call rfill_s(real(0.3, kind(q)), q)
    call v1norm(n, q, r)
    call chk('v1norm  r=sqrt(q.q) ('//"s"//')', &
             real(maxval(abs(r - abs(q)))), 1.0E-5)

    call setup9_s(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    call v3cross(n, x1, x2, x3, y1, y2, y3, w1, w2, w3)
    cx = x2*y3 - x3*y2; cy = x3*y1 - x1*y3; cz = x1*y2 - x2*y1
    call chk('v3cross w=x^y ('//"s"//')', real(max(maxval(abs(w1 - cx)), &
             max(maxval(abs(w2 - cy)), maxval(abs(w3 - cz))))), 1.0E-5)

    call setup9_s(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    call v3crossscal(n, al, x1, x2, x3, y1, y2, y3, w1, w2, w3)
    cx = al*(x2*y3 - x3*y2); cy = al*(x3*y1 - x1*y3)
    cz = al*(x1*y2 - x2*y1)
    call chk('v3crossscal w=a.(x^y) ('//"s"//')', real(max(maxval(abs(w1 - cx)), &
             max(maxval(abs(w2 - cy)), maxval(abs(w3 - cz))))), 1.0E-5)

    call setup9_s(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    call v3crossdot(n, x1, x2, x3, y1, y2, y3, w1, w2, w3, r)
    cx = x2*y3 - x3*y2; cy = x3*y1 - x1*y3; cz = x1*y2 - x2*y1
    call chk('v3crossdot r=(x^y).w ('//"s"//')', &
             real(maxval(abs(r - (cx*w1 + cy*w2 + cz*w3)))), 1.0E-5)

    call setup9_s(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    call v3crosssqr(n, x1, x2, x3, y1, y2, y3, r)
    cx = x2*y3 - x3*y2; cy = x3*y1 - x1*y3; cz = x1*y2 - x2*y1
    call chk('v3crosssqr r=(x^y).(x^y) ('//"s"//')', &
             real(maxval(abs(r - (cx*cx + cy*cy + cz*cz)))), 1.0E-5)

    call setup9_s(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    call v3crossxy_crossxz(n, x1, x2, x3, y1, y2, y3, w1, w2, w3, &
                           u1, u2, u3, v1, v2, v3)
    e1 = max(maxval(abs(u1 - (x2*y3 - x3*y2))), &
             max(maxval(abs(u2 - (x3*y1 - x1*y3))), &
                 maxval(abs(u3 - (x1*y2 - x2*y1)))))
    e2 = max(maxval(abs(v1 - (x2*w3 - x3*w2))), &
             max(maxval(abs(v2 - (x3*w1 - x1*w3))), &
                 maxval(abs(v3 - (x1*w2 - x2*w1)))))
    call chk('v3crossxy_crossxz ('//"s"//')', real(max(e1, e2)), 1.0E-5)

    call setup9_s(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    call v3crossxy_dotxz(n, x1, x2, x3, y1, y2, y3, w1, w2, w3, u1, u2, u3, r)
    e1 = max(maxval(abs(u1 - (x2*y3 - x3*y2))), &
             max(maxval(abs(u2 - (x3*y1 - x1*y3))), &
                 maxval(abs(u3 - (x1*y2 - x2*y1)))))
    call chk('v3crossxy_dotxz ('//"s"//')', &
             real(max(e1, maxval(abs(r - (x1*w1 + x2*w2 + x3*w3))))), 1.0E-5)

    call setup9_s(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    call v3dotxy_dotxz(n, x1, x2, x3, y1, y2, y3, w1, w2, w3, r, q)
    call chk('v3dotxy_dotxz ('//"s"//')', real(max(&
             maxval(abs(r - (x1*y1 + x2*y2 + x3*y3))), &
             maxval(abs(q - (x1*w1 + x2*w2 + x3*w3))))), 1.0E-5)
  end subroutine test_real_s

! Included from v3blas_f90_test.f90 with real(8) (type), TAG (name suffix),
! PK (printed precision tag) and 1.0E-12 (tolerance) defined.
! Every generic op is called through its generic name and checked against
! the defining equation recomputed with Fortran array expressions.

  subroutine rfill_d(c, v)
    real(8), intent(in) :: c
    real(8) :: v(:)
    integer :: i
    do i = 1, size(v)
       v(i) = real(i - 1, kind(v))/real(16.0, kind(v)) + c
    end do
  end subroutine rfill_d

  subroutine setup9_d(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    real(8) :: x1(:), x2(:), x3(:), y1(:), y2(:), y3(:), w1(:), w2(:), w3(:)
    real(8), parameter :: o(9) = [ 0.0, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.45, 0.5 ]
    call rfill_d(o(1), x1); call rfill_d(o(2), x2)
    call rfill_d(o(3), x3); call rfill_d(o(4), y1)
    call rfill_d(o(5), y2); call rfill_d(o(6), y3)
    call rfill_d(o(7), w1); call rfill_d(o(8), w2)
    call rfill_d(o(9), w3)
  end subroutine setup9_d

  subroutine test_real_d(n)
    integer(c_int), intent(in) :: n
    real(8) :: x1(n), x2(n), x3(n), y1(n), y2(n), y3(n)
    real(8) :: w1(n), w2(n), w3(n), u1(n), u2(n), u3(n), v1(n), v2(n), v3(n)
    real(8) :: r(n), q(n), t(n), cx(n), cy(n), cz(n), al, a2
    real(kind(u1)) :: e1, e2
    al = real(1.5, kind(al)); a2 = real(0.25, kind(a2))

    call setup9_d(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    t = y1
    call v1axpy(n, al, x1, y1)
    call chk('v1axpy  y=a*x+y ('//"d"//')', &
             real(maxval(abs(y1 - (al*x1 + t)))), 1.0E-12)

    call rfill_d(real(0.1, kind(q)), q)
    call rfill_d(real(-0.7, kind(t)), t)
    call v1xypa(n, a2, q, t, r)
    call chk('v1xypa  r=q*t+a ('//"d"//')', &
             real(maxval(abs(r - (q*t + a2)))), 1.0E-12)

    call setup9_d(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    call v3dot(n, x1, x2, x3, y1, y2, y3, r)
    call chk('v3dot   r=x.y ('//"d"//')', &
             real(maxval(abs(r - (x1*y1 + x2*y2 + x3*y3)))), 1.0E-12)

    call v3had(n, x1, x2, x3, y1, y2, y3, w1, w2, w3)
    call chk('v3had   w=x.*y ('//"d"//')', real(max(maxval(abs(w1 - x1*y1)), &
             max(maxval(abs(w2 - x2*y2)), maxval(abs(w3 - x3*y3))))), 1.0E-12)

    call v3sqr(n, x1, x2, x3, r)
    call chk('v3sqr   r=x.x ('//"d"//')', &
             real(maxval(abs(r - (x1*x1 + x2*x2 + x3*x3)))), 1.0E-12)

    call rfill_d(real(0.3, kind(q)), q)
    call v1norm(n, q, r)
    call chk('v1norm  r=sqrt(q.q) ('//"d"//')', &
             real(maxval(abs(r - abs(q)))), 1.0E-12)

    call setup9_d(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    call v3cross(n, x1, x2, x3, y1, y2, y3, w1, w2, w3)
    cx = x2*y3 - x3*y2; cy = x3*y1 - x1*y3; cz = x1*y2 - x2*y1
    call chk('v3cross w=x^y ('//"d"//')', real(max(maxval(abs(w1 - cx)), &
             max(maxval(abs(w2 - cy)), maxval(abs(w3 - cz))))), 1.0E-12)

    call setup9_d(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    call v3crossscal(n, al, x1, x2, x3, y1, y2, y3, w1, w2, w3)
    cx = al*(x2*y3 - x3*y2); cy = al*(x3*y1 - x1*y3)
    cz = al*(x1*y2 - x2*y1)
    call chk('v3crossscal w=a.(x^y) ('//"d"//')', real(max(maxval(abs(w1 - cx)), &
             max(maxval(abs(w2 - cy)), maxval(abs(w3 - cz))))), 1.0E-12)

    call setup9_d(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    call v3crossdot(n, x1, x2, x3, y1, y2, y3, w1, w2, w3, r)
    cx = x2*y3 - x3*y2; cy = x3*y1 - x1*y3; cz = x1*y2 - x2*y1
    call chk('v3crossdot r=(x^y).w ('//"d"//')', &
             real(maxval(abs(r - (cx*w1 + cy*w2 + cz*w3)))), 1.0E-12)

    call setup9_d(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    call v3crosssqr(n, x1, x2, x3, y1, y2, y3, r)
    cx = x2*y3 - x3*y2; cy = x3*y1 - x1*y3; cz = x1*y2 - x2*y1
    call chk('v3crosssqr r=(x^y).(x^y) ('//"d"//')', &
             real(maxval(abs(r - (cx*cx + cy*cy + cz*cz)))), 1.0E-12)

    call setup9_d(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    call v3crossxy_crossxz(n, x1, x2, x3, y1, y2, y3, w1, w2, w3, &
                           u1, u2, u3, v1, v2, v3)
    e1 = max(maxval(abs(u1 - (x2*y3 - x3*y2))), &
             max(maxval(abs(u2 - (x3*y1 - x1*y3))), &
                 maxval(abs(u3 - (x1*y2 - x2*y1)))))
    e2 = max(maxval(abs(v1 - (x2*w3 - x3*w2))), &
             max(maxval(abs(v2 - (x3*w1 - x1*w3))), &
                 maxval(abs(v3 - (x1*w2 - x2*w1)))))
    call chk('v3crossxy_crossxz ('//"d"//')', real(max(e1, e2)), 1.0E-12)

    call setup9_d(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    call v3crossxy_dotxz(n, x1, x2, x3, y1, y2, y3, w1, w2, w3, u1, u2, u3, r)
    e1 = max(maxval(abs(u1 - (x2*y3 - x3*y2))), &
             max(maxval(abs(u2 - (x3*y1 - x1*y3))), &
                 maxval(abs(u3 - (x1*y2 - x2*y1)))))
    call chk('v3crossxy_dotxz ('//"d"//')', &
             real(max(e1, maxval(abs(r - (x1*w1 + x2*w2 + x3*w3))))), 1.0E-12)

    call setup9_d(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    call v3dotxy_dotxz(n, x1, x2, x3, y1, y2, y3, w1, w2, w3, r, q)
    call chk('v3dotxy_dotxz ('//"d"//')', real(max(&
             maxval(abs(r - (x1*y1 + x2*y2 + x3*y3))), &
             maxval(abs(q - (x1*w1 + x2*w2 + x3*w3))))), 1.0E-12)
  end subroutine test_real_d


  subroutine cfill_c(cre, cim, v)
    complex(4) :: v(:)
    real(kind(v)), intent(in) :: cre, cim
    integer :: i
    do i = 1, size(v)
       v(i) = cmplx(real(i - 1, kind(v))/real(16.0, kind(v)) + cre, &
                    cim + real(i, kind(v))/real(64.0, kind(v)), kind(v))
    end do
  end subroutine cfill_c

  subroutine setup9c_c(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    complex(4) :: x1(:), x2(:), x3(:), y1(:), y2(:), y3(:), w1(:), w2(:), w3(:)
    real(kind(x1)) :: o(9), im(9)
    o = [ 0.0, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.45, 0.5 ]
    im = [ 0.01, 0.02, 0.03, -0.04, -0.05, -0.06, 0.07, 0.08, 0.09 ]
    call cfill_c(o(1), im(1), x1)
    call cfill_c(o(2), im(2), x2)
    call cfill_c(o(3), im(3), x3)
    call cfill_c(o(4), im(4), y1)
    call cfill_c(o(5), im(5), y2)
    call cfill_c(o(6), im(6), y3)
    call cfill_c(o(7), im(7), w1)
    call cfill_c(o(8), im(8), w2)
    call cfill_c(o(9), im(9), w3)
  end subroutine setup9c_c

  subroutine test_cplx_c(n)
    integer(c_int), intent(in) :: n
    complex(4) :: x1(n), x2(n), x3(n), y1(n), y2(n), y3(n)
    complex(4) :: w1(n), w2(n), w3(n), u1(n), u2(n), u3(n), v1(n), v2(n), v3(n)
    complex(4) :: r(n), q(n), t(n), cx(n), cy(n), cz(n), al, a2
    real(kind(u1)) :: e1, e2
    al = cmplx(real(1.5, kind(al)), real(0.5, kind(al)), kind(al))
    a2 = cmplx(real(0.25, kind(a2)), real(0.1, kind(a2)), kind(a2))

    call setup9c_c(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    t = y1
    call v1axpy(n, al, x1, y1)
    call chk('v1axpy  y=a*x+y ('//"c"//')', &
             real(maxval(abs(y1 - (al*x1 + t)))), 1.0E-5)

    call cfill_c(real(0.1, kind(q)), real(0.05, kind(q)), q)
    call cfill_c(real(-0.7, kind(t)), real(0.3, kind(t)), t)
    call v1xypa(n, a2, q, t, r)
    call chk('v1xypa  r=q*t+a ('//"c"//')', &
             real(maxval(abs(r - (q*t + a2)))), 1.0E-5)

    call setup9c_c(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    call v3dot(n, x1, x2, x3, y1, y2, y3, r)
    call chk('v3dot   r=x.y ('//"c"//')', &
             real(maxval(abs(r - (x1*y1 + x2*y2 + x3*y3)))), 1.0E-5)

    call v3had(n, x1, x2, x3, y1, y2, y3, w1, w2, w3)
    call chk('v3had   w=x.*y ('//"c"//')', real(max(maxval(abs(w1 - x1*y1)), &
             max(maxval(abs(w2 - x2*y2)), maxval(abs(w3 - x3*y3))))), 1.0E-5)

    call v3sqr(n, x1, x2, x3, r)
    call chk('v3sqr   r=x.x ('//"c"//')', &
             real(maxval(abs(r - (x1*x1 + x2*x2 + x3*x3)))), 1.0E-5)

    call cfill_c(real(0.3, kind(q)), real(-0.2, kind(q)), q)
    call v1norm(n, q, r)
    call chk('v1norm  r=sqrt(q.q) ('//"c"//')', &
             real(maxval(abs(r - sqrt(q*q)))), 1.0E-5)

    call setup9c_c(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    call v3cross(n, x1, x2, x3, y1, y2, y3, w1, w2, w3)
    cx = x2*y3 - x3*y2; cy = x3*y1 - x1*y3; cz = x1*y2 - x2*y1
    call chk('v3cross w=x^y ('//"c"//')', real(max(maxval(abs(w1 - cx)), &
             max(maxval(abs(w2 - cy)), maxval(abs(w3 - cz))))), 1.0E-5)

    call setup9c_c(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    call v3crossscal(n, al, x1, x2, x3, y1, y2, y3, w1, w2, w3)
    cx = al*(x2*y3 - x3*y2); cy = al*(x3*y1 - x1*y3)
    cz = al*(x1*y2 - x2*y1)
    call chk('v3crossscal w=a.(x^y) ('//"c"//')', real(max(maxval(abs(w1 - cx)), &
             max(maxval(abs(w2 - cy)), maxval(abs(w3 - cz))))), 1.0E-5)

    call setup9c_c(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    call v3crossdot(n, x1, x2, x3, y1, y2, y3, w1, w2, w3, r)
    cx = x2*y3 - x3*y2; cy = x3*y1 - x1*y3; cz = x1*y2 - x2*y1
    call chk('v3crossdot r=(x^y).w ('//"c"//')', &
             real(maxval(abs(r - (cx*w1 + cy*w2 + cz*w3)))), 1.0E-5)

    call setup9c_c(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    call v3crosssqr(n, x1, x2, x3, y1, y2, y3, r)
    cx = x2*y3 - x3*y2; cy = x3*y1 - x1*y3; cz = x1*y2 - x2*y1
    call chk('v3crosssqr r=(x^y).(x^y) ('//"c"//')', &
             real(maxval(abs(r - (cx*cx + cy*cy + cz*cz)))), 1.0E-5)

    call setup9c_c(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    call v3crossxy_crossxz(n, x1, x2, x3, y1, y2, y3, w1, w2, w3, &
                           u1, u2, u3, v1, v2, v3)
    e1 = max(maxval(abs(u1 - (x2*y3 - x3*y2))), &
             max(maxval(abs(u2 - (x3*y1 - x1*y3))), &
                 maxval(abs(u3 - (x1*y2 - x2*y1)))))
    e2 = max(maxval(abs(v1 - (x2*w3 - x3*w2))), &
             max(maxval(abs(v2 - (x3*w1 - x1*w3))), &
                 maxval(abs(v3 - (x1*w2 - x2*w1)))))
    call chk('v3crossxy_crossxz ('//"c"//')', real(max(e1, e2)), 1.0E-5)

    call setup9c_c(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    call v3crossxy_dotxz(n, x1, x2, x3, y1, y2, y3, w1, w2, w3, u1, u2, u3, r)
    e1 = max(maxval(abs(u1 - (x2*y3 - x3*y2))), &
             max(maxval(abs(u2 - (x3*y1 - x1*y3))), &
                 maxval(abs(u3 - (x1*y2 - x2*y1)))))
    call chk('v3crossxy_dotxz ('//"c"//')', &
             real(max(e1, maxval(abs(r - (x1*w1 + x2*w2 + x3*w3))))), 1.0E-5)

    call setup9c_c(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    call v3dotxy_dotxz(n, x1, x2, x3, y1, y2, y3, w1, w2, w3, r, q)
    call chk('v3dotxy_dotxz ('//"c"//')', real(max(&
             maxval(abs(r - (x1*y1 + x2*y2 + x3*y3))), &
             maxval(abs(q - (x1*w1 + x2*w2 + x3*w3))))), 1.0E-5)
  end subroutine test_cplx_c


  subroutine cfill_z(cre, cim, v)
    complex(8) :: v(:)
    real(kind(v)), intent(in) :: cre, cim
    integer :: i
    do i = 1, size(v)
       v(i) = cmplx(real(i - 1, kind(v))/real(16.0, kind(v)) + cre, &
                    cim + real(i, kind(v))/real(64.0, kind(v)), kind(v))
    end do
  end subroutine cfill_z

  subroutine setup9c_z(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    complex(8) :: x1(:), x2(:), x3(:), y1(:), y2(:), y3(:), w1(:), w2(:), w3(:)
    real(kind(x1)) :: o(9), im(9)
    o = [ 0.0, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.45, 0.5 ]
    im = [ 0.01, 0.02, 0.03, -0.04, -0.05, -0.06, 0.07, 0.08, 0.09 ]
    call cfill_z(o(1), im(1), x1)
    call cfill_z(o(2), im(2), x2)
    call cfill_z(o(3), im(3), x3)
    call cfill_z(o(4), im(4), y1)
    call cfill_z(o(5), im(5), y2)
    call cfill_z(o(6), im(6), y3)
    call cfill_z(o(7), im(7), w1)
    call cfill_z(o(8), im(8), w2)
    call cfill_z(o(9), im(9), w3)
  end subroutine setup9c_z

  subroutine test_cplx_z(n)
    integer(c_int), intent(in) :: n
    complex(8) :: x1(n), x2(n), x3(n), y1(n), y2(n), y3(n)
    complex(8) :: w1(n), w2(n), w3(n), u1(n), u2(n), u3(n), v1(n), v2(n), v3(n)
    complex(8) :: r(n), q(n), t(n), cx(n), cy(n), cz(n), al, a2
    real(kind(u1)) :: e1, e2
    al = cmplx(real(1.5, kind(al)), real(0.5, kind(al)), kind(al))
    a2 = cmplx(real(0.25, kind(a2)), real(0.1, kind(a2)), kind(a2))

    call setup9c_z(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    t = y1
    call v1axpy(n, al, x1, y1)
    call chk('v1axpy  y=a*x+y ('//"z"//')', &
             real(maxval(abs(y1 - (al*x1 + t)))), 1.0E-12)

    call cfill_z(real(0.1, kind(q)), real(0.05, kind(q)), q)
    call cfill_z(real(-0.7, kind(t)), real(0.3, kind(t)), t)
    call v1xypa(n, a2, q, t, r)
    call chk('v1xypa  r=q*t+a ('//"z"//')', &
             real(maxval(abs(r - (q*t + a2)))), 1.0E-12)

    call setup9c_z(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    call v3dot(n, x1, x2, x3, y1, y2, y3, r)
    call chk('v3dot   r=x.y ('//"z"//')', &
             real(maxval(abs(r - (x1*y1 + x2*y2 + x3*y3)))), 1.0E-12)

    call v3had(n, x1, x2, x3, y1, y2, y3, w1, w2, w3)
    call chk('v3had   w=x.*y ('//"z"//')', real(max(maxval(abs(w1 - x1*y1)), &
             max(maxval(abs(w2 - x2*y2)), maxval(abs(w3 - x3*y3))))), 1.0E-12)

    call v3sqr(n, x1, x2, x3, r)
    call chk('v3sqr   r=x.x ('//"z"//')', &
             real(maxval(abs(r - (x1*x1 + x2*x2 + x3*x3)))), 1.0E-12)

    call cfill_z(real(0.3, kind(q)), real(-0.2, kind(q)), q)
    call v1norm(n, q, r)
    call chk('v1norm  r=sqrt(q.q) ('//"z"//')', &
             real(maxval(abs(r - sqrt(q*q)))), 1.0E-12)

    call setup9c_z(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    call v3cross(n, x1, x2, x3, y1, y2, y3, w1, w2, w3)
    cx = x2*y3 - x3*y2; cy = x3*y1 - x1*y3; cz = x1*y2 - x2*y1
    call chk('v3cross w=x^y ('//"z"//')', real(max(maxval(abs(w1 - cx)), &
             max(maxval(abs(w2 - cy)), maxval(abs(w3 - cz))))), 1.0E-12)

    call setup9c_z(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    call v3crossscal(n, al, x1, x2, x3, y1, y2, y3, w1, w2, w3)
    cx = al*(x2*y3 - x3*y2); cy = al*(x3*y1 - x1*y3)
    cz = al*(x1*y2 - x2*y1)
    call chk('v3crossscal w=a.(x^y) ('//"z"//')', real(max(maxval(abs(w1 - cx)), &
             max(maxval(abs(w2 - cy)), maxval(abs(w3 - cz))))), 1.0E-12)

    call setup9c_z(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    call v3crossdot(n, x1, x2, x3, y1, y2, y3, w1, w2, w3, r)
    cx = x2*y3 - x3*y2; cy = x3*y1 - x1*y3; cz = x1*y2 - x2*y1
    call chk('v3crossdot r=(x^y).w ('//"z"//')', &
             real(maxval(abs(r - (cx*w1 + cy*w2 + cz*w3)))), 1.0E-12)

    call setup9c_z(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    call v3crosssqr(n, x1, x2, x3, y1, y2, y3, r)
    cx = x2*y3 - x3*y2; cy = x3*y1 - x1*y3; cz = x1*y2 - x2*y1
    call chk('v3crosssqr r=(x^y).(x^y) ('//"z"//')', &
             real(maxval(abs(r - (cx*cx + cy*cy + cz*cz)))), 1.0E-12)

    call setup9c_z(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    call v3crossxy_crossxz(n, x1, x2, x3, y1, y2, y3, w1, w2, w3, &
                           u1, u2, u3, v1, v2, v3)
    e1 = max(maxval(abs(u1 - (x2*y3 - x3*y2))), &
             max(maxval(abs(u2 - (x3*y1 - x1*y3))), &
                 maxval(abs(u3 - (x1*y2 - x2*y1)))))
    e2 = max(maxval(abs(v1 - (x2*w3 - x3*w2))), &
             max(maxval(abs(v2 - (x3*w1 - x1*w3))), &
                 maxval(abs(v3 - (x1*w2 - x2*w1)))))
    call chk('v3crossxy_crossxz ('//"z"//')', real(max(e1, e2)), 1.0E-12)

    call setup9c_z(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    call v3crossxy_dotxz(n, x1, x2, x3, y1, y2, y3, w1, w2, w3, u1, u2, u3, r)
    e1 = max(maxval(abs(u1 - (x2*y3 - x3*y2))), &
             max(maxval(abs(u2 - (x3*y1 - x1*y3))), &
                 maxval(abs(u3 - (x1*y2 - x2*y1)))))
    call chk('v3crossxy_dotxz ('//"z"//')', &
             real(max(e1, maxval(abs(r - (x1*w1 + x2*w2 + x3*w3))))), 1.0E-12)

    call setup9c_z(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    call v3dotxy_dotxz(n, x1, x2, x3, y1, y2, y3, w1, w2, w3, r, q)
    call chk('v3dotxy_dotxz ('//"z"//')', real(max(&
             maxval(abs(r - (x1*y1 + x2*y2 + x3*y3))), &
             maxval(abs(q - (x1*w1 + x2*w2 + x3*w3))))), 1.0E-12)
  end subroutine test_cplx_z
end module v3blas_f90_cases

program v3blas_f90_test
  use v3blas_f90_cases
  use, intrinsic :: iso_c_binding, only: c_int
  implicit none
  integer(c_int), parameter :: n = 8

  call test_real_s(n)
  call test_real_d(n)
  call test_cplx_c(n)
  call test_cplx_z(n)

  if (fails == 0) then
     write(*, '(a)') 'v3blas_f90_test: OK (0 failures)'
  else
     write(*, '(a, i0, a)') 'v3blas_f90_test: FAIL (', fails, ' failures)'
     stop 1
  end if
end program v3blas_f90_test
