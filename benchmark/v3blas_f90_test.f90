! v3blas_f90_test.f90 - correctness test for the v3blas Fortran front-end.
!
! All 18 operations are called through their generic names at all four
! precisions (the compiler picks s/d/c/z from the argument types).  A
! 3-vector is one field array u(n, 3): column c is component c, so the
! columns are exactly the unit-stride component arrays the kernels read.
! The one-vector ops (v1xypa, v1norm) keep the explicit-n form.
! Results are checked against the defining equation recomputed with
! Fortran array expressions over the same fields.
!
! Usage: ./v3blas_f90_test [s|d|c|z]     (default: all four precisions)
! Build: gfortran -Wall ../openblas/v3blas.f90 v3blas_f90_test.f90 \
!        -L../pkgs/openblas/lib -lopenblas -lm -o v3blas_f90_test

module v3blas_f90_cases
  use v3blas
  use, intrinsic :: iso_c_binding, only: c_int
  implicit none
  integer, public :: fails = 0
  private :: rfill_s, rfill_d, setup9_s, setup9_d
  private :: pack_s, pack_d, cr_s, cr_d, rd_s, rd_d
  private :: cfill_c, cfill_z, setup9c_c, setup9c_z
  private :: pack_c, pack_z, cr_c, cr_z, rd_c, rd_z
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

  ! ---- data helpers ------------------------------------------------------
  ! pack: component arrays -> field columns.  cr/rd: cross and dot of two
  ! fields, i.e. the defining equations written the way users see them.

  subroutine rfill_s(c, v)
    real(4), intent(in) :: c
    real(4) :: v(:)
    integer :: i
    do i = 1, size(v)
       v(i) = real(i - 1, kind(v))/real(16.0, kind(v)) + c
    end do
  end subroutine rfill_s

  subroutine rfill_d(c, v)
    real(8), intent(in) :: c
    real(8) :: v(:)
    integer :: i
    do i = 1, size(v)
       v(i) = real(i - 1, kind(v))/real(16.0d0, kind(v)) + c
    end do
  end subroutine rfill_d

  subroutine setup9_s(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    real(4) :: x1(:), x2(:), x3(:), y1(:), y2(:), y3(:), w1(:), w2(:), w3(:)
    real(4), parameter :: o(9) = [ 0.0, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.45, 0.5 ]
    call rfill_s(o(1), x1); call rfill_s(o(2), x2)
    call rfill_s(o(3), x3); call rfill_s(o(4), y1)
    call rfill_s(o(5), y2); call rfill_s(o(6), y3)
    call rfill_s(o(7), w1); call rfill_s(o(8), w2)
    call rfill_s(o(9), w3)
  end subroutine setup9_s

  subroutine setup9_d(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    real(8) :: x1(:), x2(:), x3(:), y1(:), y2(:), y3(:), w1(:), w2(:), w3(:)
    real(8), parameter :: o(9) = [ 0.d0, 0.05d0, 0.1d0, 0.2d0, 0.25d0, &
                                   0.3d0, 0.4d0, 0.45d0, 0.5d0 ]
    call rfill_d(o(1), x1); call rfill_d(o(2), x2)
    call rfill_d(o(3), x3); call rfill_d(o(4), y1)
    call rfill_d(o(5), y2); call rfill_d(o(6), y3)
    call rfill_d(o(7), w1); call rfill_d(o(8), w2)
    call rfill_d(o(9), w3)
  end subroutine setup9_d

  subroutine cfill_c(re, im, v)
    real, intent(in) :: re, im
    complex(4) :: v(:)
    integer :: i
    do i = 1, size(v)
       v(i) = cmplx(re + real(i - 1)/16.0, im - real(i - 1)/32.0, kind(v))
    end do
  end subroutine cfill_c

  subroutine cfill_z(re, im, v)
    real(8), intent(in) :: re, im
    complex(8) :: v(:)
    integer :: i
    do i = 1, size(v)
       v(i) = cmplx(re + real(i - 1, 8)/16.0d0, im - real(i - 1, 8)/32.0d0, kind(v))
    end do
  end subroutine cfill_z

  subroutine setup9c_c(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    complex(4) :: x1(:), x2(:), x3(:), y1(:), y2(:), y3(:), w1(:), w2(:), w3(:)
    call cfill_c(0.0, 0.10, x1); call cfill_c(0.05, -0.10, x2)
    call cfill_c(0.10, 0.05, x3); call cfill_c(0.20, -0.05, y1)
    call cfill_c(0.25, 0.15, y2); call cfill_c(0.30, -0.15, y3)
    call cfill_c(0.40, 0.05, w1); call cfill_c(0.45, -0.20, w2)
    call cfill_c(0.50, 0.20, w3)
  end subroutine setup9c_c

  subroutine setup9c_z(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    complex(8) :: x1(:), x2(:), x3(:), y1(:), y2(:), y3(:), w1(:), w2(:), w3(:)
    call cfill_z(0.d0, 0.10d0, x1); call cfill_z(0.05d0, -0.10d0, x2)
    call cfill_z(0.10d0, 0.05d0, x3); call cfill_z(0.20d0, -0.05d0, y1)
    call cfill_z(0.25d0, 0.15d0, y2); call cfill_z(0.30d0, -0.15d0, y3)
    call cfill_z(0.40d0, 0.05d0, w1); call cfill_z(0.45d0, -0.20d0, w2)
    call cfill_z(0.50d0, 0.20d0, w3)
  end subroutine setup9c_z

  subroutine pack_s(a1, a2, a3, a)
    real(4) :: a1(:), a2(:), a3(:), a(:, :)
    a(:, 1) = a1; a(:, 2) = a2; a(:, 3) = a3
  end subroutine pack_s

  subroutine pack_d(a1, a2, a3, a)
    real(8) :: a1(:), a2(:), a3(:), a(:, :)
    a(:, 1) = a1; a(:, 2) = a2; a(:, 3) = a3
  end subroutine pack_d

  subroutine pack_c(a1, a2, a3, a)
    complex(4) :: a1(:), a2(:), a3(:), a(:, :)
    a(:, 1) = a1; a(:, 2) = a2; a(:, 3) = a3
  end subroutine pack_c

  subroutine pack_z(a1, a2, a3, a)
    complex(8) :: a1(:), a2(:), a3(:), a(:, :)
    a(:, 1) = a1; a(:, 2) = a2; a(:, 3) = a3
  end subroutine pack_z

  function cr_s(a, b) result(c)
    real(4) :: a(:, :), b(:, :), c(size(a, 1), 3)
    c(:, 1) = a(:, 2)*b(:, 3) - a(:, 3)*b(:, 2)
    c(:, 2) = a(:, 3)*b(:, 1) - a(:, 1)*b(:, 3)
    c(:, 3) = a(:, 1)*b(:, 2) - a(:, 2)*b(:, 1)
  end function cr_s

  function cr_d(a, b) result(c)
    real(8) :: a(:, :), b(:, :), c(size(a, 1), 3)
    c(:, 1) = a(:, 2)*b(:, 3) - a(:, 3)*b(:, 2)
    c(:, 2) = a(:, 3)*b(:, 1) - a(:, 1)*b(:, 3)
    c(:, 3) = a(:, 1)*b(:, 2) - a(:, 2)*b(:, 1)
  end function cr_d

  function cr_c(a, b) result(c)
    complex(4) :: a(:, :), b(:, :), c(size(a, 1), 3)
    c(:, 1) = a(:, 2)*b(:, 3) - a(:, 3)*b(:, 2)
    c(:, 2) = a(:, 3)*b(:, 1) - a(:, 1)*b(:, 3)
    c(:, 3) = a(:, 1)*b(:, 2) - a(:, 2)*b(:, 1)
  end function cr_c

  function cr_z(a, b) result(c)
    complex(8) :: a(:, :), b(:, :), c(size(a, 1), 3)
    c(:, 1) = a(:, 2)*b(:, 3) - a(:, 3)*b(:, 2)
    c(:, 2) = a(:, 3)*b(:, 1) - a(:, 1)*b(:, 3)
    c(:, 3) = a(:, 1)*b(:, 2) - a(:, 2)*b(:, 1)
  end function cr_z

  function rd_s(a, b) result(r)
    real(4) :: a(:, :), b(:, :), r(size(a, 1))
    r = sum(a*b, dim=2)
  end function rd_s

  function rd_d(a, b) result(r)
    real(8) :: a(:, :), b(:, :), r(size(a, 1))
    r = sum(a*b, dim=2)
  end function rd_d

  function rd_c(a, b) result(r)
    complex(4) :: a(:, :), b(:, :), r(size(a, 1))
    r = sum(a*b, dim=2)
  end function rd_c

  function rd_z(a, b) result(r)
    complex(8) :: a(:, :), b(:, :), r(size(a, 1))
    r = sum(a*b, dim=2)
  end function rd_z

  ! ---- real single -------------------------------------------------------

  subroutine test_real_s(n)
    integer(c_int), intent(in) :: n
    real(4) :: x1(n), x2(n), x3(n), y1(n), y2(n), y3(n)
    real(4) :: w1(n), w2(n), w3(n)
    real(4) :: xf(n, 3), yf(n, 3), wf(n, 3), uf(n, 3), vf(n, 3), rf(n, 3)
    real(4) :: r(n), q(n), t(n), al, a2, eps
    al = 1.5; a2 = 0.25; eps = 0.25

    call rfill_s(0.1, q); call rfill_s(-0.7, t)
    call v1xypa(n, q, t, a2, r)
    call chk('v1xypa  r=q*t+a ('//"s"//')', &
             real(maxval(abs(r - (q*t + a2)))), 1.0E-5)

    call setup9_s(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    call pack_s(x1, x2, x3, xf); call pack_s(y1, y2, y3, yf)
    call pack_s(w1, w2, w3, wf)

    call v3dot(xf, yf, r)
    call chk('v3dot   r=x.y ('//"s"//')', &
             real(maxval(abs(r - rd_s(xf, yf)))), 1.0E-5)

    call v3had(xf, yf, wf)
    call chk('v3had   w=x.*y ('//"s"//')', &
             real(maxval(abs(wf - xf*yf))), 1.0E-5)

    call v3sqr(xf, r)
    call chk('v3sqr   r=x.x ('//"s"//')', &
             real(maxval(abs(r - rd_s(xf, xf)))), 1.0E-5)

    call rfill_s(0.3, q)
    call v1norm(n, q, r)
    call chk('v1norm  r=sqrt(q.q) ('//"s"//')', &
             real(maxval(abs(r - abs(q)))), 1.0E-5)

    call v3cross(xf, yf, wf)
    call chk('v3cross w=x^y ('//"s"//')', &
             real(maxval(abs(wf - cr_s(xf, yf)))), 1.0E-5)

    call v3crossscal(xf, yf, al, wf)
    call chk('v3crossscal w=a.(x^y) ('//"s"//')', &
             real(maxval(abs(wf - al*cr_s(xf, yf)))), 1.0E-5)

    call v3crossdot(xf, yf, wf, r)
    call chk('v3crossdot r=(x^y).w ('//"s"//')', &
             real(maxval(abs(r - rd_s(cr_s(xf, yf), wf)))), 1.0E-5)

    call v3crosssqr(xf, yf, r)
    call chk('v3crosssqr r=(x^y).(x^y) ('//"s"//')', &
             real(maxval(abs(r - rd_s(cr_s(xf, yf), cr_s(xf, yf))))), 1.0E-5)

    call v3crossxy_crossxz(xf, yf, wf, uf, vf)
    call chk('v3crossxy_crossxz ('//"s"//')', real(max(&
             maxval(abs(uf - cr_s(xf, yf))), &
             maxval(abs(vf - cr_s(xf, wf))))), 1.0E-5)

    call v3crossxy_dotxz(xf, yf, wf, uf, r)
    call chk('v3crossxy_dotxz ('//"s"//')', real(max(&
             maxval(abs(uf - cr_s(xf, yf))), &
             maxval(abs(r - rd_s(xf, wf))))), 1.0E-5)

    call v3dotxy_dotxz(xf, yf, wf, r, q)
    call chk('v3dotxy_dotxz ('//"s"//')', real(max(&
             maxval(abs(r - rd_s(xf, yf))), &
             maxval(abs(q - rd_s(xf, wf))))), 1.0E-5)

    call v3crosscross(xf, yf, wf, rf)
    call chk('v3crosscross (s)', &
             real(maxval(abs(rf - cr_s(cr_s(xf, yf), wf)))), 1.0E-5)

    call v3norm_unit(xf, eps, r, uf)
    q = sqrt(sum(xf*xf, dim=2) + eps)
    call chk('v3norm_unit (s)', real(max(maxval(abs(r - q)), &
             maxval(abs(uf - xf/spread(q, 2, 3))))), 1.0E-5)

    call v3refl(xf, yf, al, eps, rf)
    q = al*rd_s(xf, yf)/(rd_s(yf, yf) + eps)
    call chk('v3refl (s)', real(maxval(abs(rf - &
             (xf - spread(q, 2, 3)*yf)))), 1.0E-5)

    call v3exb(xf, yf, al, eps, rf)
    q = rd_s(yf, yf) + eps
    call chk('v3exb (s)', real(maxval(abs(rf - &
             al*cr_s(xf, yf)/spread(q, 2, 3)))), 1.0E-5)

    call v3drag(xf, al, eps, rf)
    q = sqrt(sum(xf*xf, dim=2) + eps)
    call chk('v3drag (s)', real(maxval(abs(rf - &
             al*xf*spread(q, 2, 3)))), 1.0E-5)

    call rfill_s(0.3, t)
    call v3mom_ke(t, xf, uf, q)
    call chk('v3mom_ke (s)', real(max(&
             maxval(abs(uf - spread(t, 2, 3)*xf)), &
             maxval(abs(q - real(0.5)*t*rd_s(xf, xf))))), 1.0E-5)
  end subroutine test_real_s

  ! ---- real double -------------------------------------------------------

  subroutine test_real_d(n)
    integer(c_int), intent(in) :: n
    real(8) :: x1(n), x2(n), x3(n), y1(n), y2(n), y3(n)
    real(8) :: w1(n), w2(n), w3(n)
    real(8) :: xf(n, 3), yf(n, 3), wf(n, 3), uf(n, 3), vf(n, 3), rf(n, 3)
    real(8) :: r(n), q(n), t(n), al, a2, eps
    al = 1.5d0; a2 = 0.25d0; eps = 0.25d0

    call rfill_d(0.1d0, q); call rfill_d(-0.7d0, t)
    call v1xypa(n, q, t, a2, r)
    call chk('v1xypa  r=q*t+a ('//"d"//')', &
             real(maxval(abs(r - (q*t + a2)))), 1.0E-12)

    call setup9_d(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    call pack_d(x1, x2, x3, xf); call pack_d(y1, y2, y3, yf)
    call pack_d(w1, w2, w3, wf)

    call v3dot(xf, yf, r)
    call chk('v3dot   r=x.y ('//"d"//')', &
             real(maxval(abs(r - rd_d(xf, yf)))), 1.0E-12)

    call v3had(xf, yf, wf)
    call chk('v3had   w=x.*y ('//"d"//')', &
             real(maxval(abs(wf - xf*yf))), 1.0E-12)

    call v3sqr(xf, r)
    call chk('v3sqr   r=x.x ('//"d"//')', &
             real(maxval(abs(r - rd_d(xf, xf)))), 1.0E-12)

    call rfill_d(0.3d0, q)
    call v1norm(n, q, r)
    call chk('v1norm  r=sqrt(q.q) ('//"d"//')', &
             real(maxval(abs(r - abs(q)))), 1.0E-12)

    call v3cross(xf, yf, wf)
    call chk('v3cross w=x^y ('//"d"//')', &
             real(maxval(abs(wf - cr_d(xf, yf)))), 1.0E-12)

    call v3crossscal(xf, yf, al, wf)
    call chk('v3crossscal w=a.(x^y) ('//"d"//')', &
             real(maxval(abs(wf - al*cr_d(xf, yf)))), 1.0E-12)

    call v3crossdot(xf, yf, wf, r)
    call chk('v3crossdot r=(x^y).w ('//"d"//')', &
             real(maxval(abs(r - rd_d(cr_d(xf, yf), wf)))), 1.0E-12)

    call v3crosssqr(xf, yf, r)
    call chk('v3crosssqr r=(x^y).(x^y) ('//"d"//')', &
             real(maxval(abs(r - rd_d(cr_d(xf, yf), cr_d(xf, yf))))), 1.0E-12)

    call v3crossxy_crossxz(xf, yf, wf, uf, vf)
    call chk('v3crossxy_crossxz ('//"d"//')', real(max(&
             maxval(abs(uf - cr_d(xf, yf))), &
             maxval(abs(vf - cr_d(xf, wf))))), 1.0E-12)

    call v3crossxy_dotxz(xf, yf, wf, uf, r)
    call chk('v3crossxy_dotxz ('//"d"//')', real(max(&
             maxval(abs(uf - cr_d(xf, yf))), &
             maxval(abs(r - rd_d(xf, wf))))), 1.0E-12)

    call v3dotxy_dotxz(xf, yf, wf, r, q)
    call chk('v3dotxy_dotxz ('//"d"//')', real(max(&
             maxval(abs(r - rd_d(xf, yf))), &
             maxval(abs(q - rd_d(xf, wf))))), 1.0E-12)

    call v3crosscross(xf, yf, wf, rf)
    call chk('v3crosscross (d)', &
             real(maxval(abs(rf - cr_d(cr_d(xf, yf), wf)))), 1.0E-12)

    call v3norm_unit(xf, eps, r, uf)
    q = sqrt(sum(xf*xf, dim=2) + eps)
    call chk('v3norm_unit (d)', real(max(maxval(abs(r - q)), &
             maxval(abs(uf - xf/spread(q, 2, 3))))), 1.0E-12)

    call v3refl(xf, yf, al, eps, rf)
    q = al*rd_d(xf, yf)/(rd_d(yf, yf) + eps)
    call chk('v3refl (d)', real(maxval(abs(rf - &
             (xf - spread(q, 2, 3)*yf)))), 1.0E-12)

    call v3exb(xf, yf, al, eps, rf)
    q = rd_d(yf, yf) + eps
    call chk('v3exb (d)', real(maxval(abs(rf - &
             al*cr_d(xf, yf)/spread(q, 2, 3)))), 1.0E-12)

    call v3drag(xf, al, eps, rf)
    q = sqrt(sum(xf*xf, dim=2) + eps)
    call chk('v3drag (d)', real(maxval(abs(rf - &
             al*xf*spread(q, 2, 3)))), 1.0E-12)

    call rfill_d(0.3d0, t)
    call v3mom_ke(t, xf, uf, q)
    call chk('v3mom_ke (d)', real(max(&
             maxval(abs(uf - spread(t, 2, 3)*xf)), &
             maxval(abs(q - 0.5d0*t*rd_d(xf, xf))))), 1.0E-12)
  end subroutine test_real_d

  ! ---- complex single (dot is unconjugated; scalars a, eps are real) ----

  subroutine test_cplx_c(n)
    integer(c_int), intent(in) :: n
    complex(4) :: x1(n), x2(n), x3(n), y1(n), y2(n), y3(n)
    complex(4) :: w1(n), w2(n), w3(n)
    complex(4) :: xf(n, 3), yf(n, 3), wf(n, 3), uf(n, 3), vf(n, 3), rf(n, 3)
    complex(4) :: r(n), q(n), t(n), al, a2, h
    real(4) :: eps
    al = cmplx(1.5, 0.5); a2 = cmplx(0.25, 0.1); h = cmplx(0.5, 0.0)
    eps = 0.25

    call cfill_c(0.1, 0.05, q); call cfill_c(-0.7, 0.3, t)
    call v1xypa(n, q, t, a2, r)
    call chk('v1xypa  r=q*t+a ('//"c"//')', &
             real(maxval(abs(r - (q*t + a2)))), 1.0E-5)

    call setup9c_c(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    call pack_c(x1, x2, x3, xf); call pack_c(y1, y2, y3, yf)
    call pack_c(w1, w2, w3, wf)

    call v3dot(xf, yf, r)
    call chk('v3dot   r=x.y ('//"c"//')', &
             real(maxval(abs(r - rd_c(xf, yf)))), 1.0E-5)

    call v3had(xf, yf, wf)
    call chk('v3had   w=x.*y ('//"c"//')', &
             real(maxval(abs(wf - xf*yf))), 1.0E-5)

    call v3sqr(xf, r)
    call chk('v3sqr   r=x.x ('//"c"//')', &
             real(maxval(abs(r - rd_c(xf, xf)))), 1.0E-5)

    call cfill_c(0.3, -0.2, q)
    call v1norm(n, q, r)
    call chk('v1norm  r=sqrt(q.q) ('//"c"//')', &
             real(maxval(abs(r - sqrt(q*q)))), 1.0E-5)

    call v3cross(xf, yf, wf)
    call chk('v3cross w=x^y ('//"c"//')', &
             real(maxval(abs(wf - cr_c(xf, yf)))), 1.0E-5)

    call v3crossscal(xf, yf, al, wf)
    call chk('v3crossscal w=a.(x^y) ('//"c"//')', &
             real(maxval(abs(wf - al*cr_c(xf, yf)))), 1.0E-5)

    call v3crossdot(xf, yf, wf, r)
    call chk('v3crossdot r=(x^y).w ('//"c"//')', &
             real(maxval(abs(r - rd_c(cr_c(xf, yf), wf)))), 1.0E-5)

    call v3crosssqr(xf, yf, r)
    call chk('v3crosssqr r=(x^y).(x^y) ('//"c"//')', &
             real(maxval(abs(r - rd_c(cr_c(xf, yf), cr_c(xf, yf))))), 1.0E-5)

    call v3crossxy_crossxz(xf, yf, wf, uf, vf)
    call chk('v3crossxy_crossxz ('//"c"//')', real(max(&
             maxval(abs(uf - cr_c(xf, yf))), &
             maxval(abs(vf - cr_c(xf, wf))))), 1.0E-5)

    call v3crossxy_dotxz(xf, yf, wf, uf, r)
    call chk('v3crossxy_dotxz ('//"c"//')', real(max(&
             maxval(abs(uf - cr_c(xf, yf))), &
             maxval(abs(r - rd_c(xf, wf))))), 1.0E-5)

    call v3dotxy_dotxz(xf, yf, wf, r, q)
    call chk('v3dotxy_dotxz ('//"c"//')', real(max(&
             maxval(abs(r - rd_c(xf, yf))), &
             maxval(abs(q - rd_c(xf, wf))))), 1.0E-5)

    call v3crosscross(xf, yf, wf, rf)
    call chk('v3crosscross (c)', &
             real(maxval(abs(rf - cr_c(cr_c(xf, yf), wf)))), 1.0E-5)

    call v3norm_unit(xf, eps, r, uf)
    q = sqrt(sum(xf*xf, dim=2) + eps)
    call chk('v3norm_unit (c)', real(max(maxval(abs(r - q)), &
             maxval(abs(uf - xf/spread(q, 2, 3))))), 1.0E-5)

    call v3refl(xf, yf, 1.5, eps, rf)
    q = cmplx(1.5, 0.0)*rd_c(xf, yf)/(rd_c(yf, yf) + eps)
    call chk('v3refl (c)', real(maxval(abs(rf - &
             (xf - spread(q, 2, 3)*yf)))), 1.0E-5)

    call v3exb(xf, yf, 1.5, eps, rf)
    q = rd_c(yf, yf) + eps
    call chk('v3exb (c)', real(maxval(abs(rf - &
             cmplx(1.5, 0.0)*cr_c(xf, yf)/spread(q, 2, 3)))), 1.0E-5)

    call v3drag(xf, 1.5, eps, rf)
    q = sqrt(sum(xf*xf, dim=2) + eps)
    call chk('v3drag (c)', real(maxval(abs(rf - &
             cmplx(1.5, 0.0)*xf*spread(q, 2, 3)))), 1.0E-5)

    call cfill_c(0.3, -0.2, t)
    call v3mom_ke(t, xf, uf, q)
    call chk('v3mom_ke (c)', real(max(&
             maxval(abs(uf - spread(t, 2, 3)*xf)), &
             maxval(abs(q - h*t*rd_c(xf, xf))))), 1.0E-5)
  end subroutine test_cplx_c

  ! ---- complex double -----------------------------------------------------

  subroutine test_cplx_z(n)
    integer(c_int), intent(in) :: n
    complex(8) :: x1(n), x2(n), x3(n), y1(n), y2(n), y3(n)
    complex(8) :: w1(n), w2(n), w3(n)
    complex(8) :: xf(n, 3), yf(n, 3), wf(n, 3), uf(n, 3), vf(n, 3), rf(n, 3)
    complex(8) :: r(n), q(n), t(n), al, a2, h
    real(8) :: eps
    al = cmplx(1.5d0, 0.5d0, kind=8); a2 = cmplx(0.25d0, 0.1d0, kind=8); h = cmplx(0.5d0, 0.d0, kind=8)
    eps = 0.25d0

    call cfill_z(0.1d0, 0.05d0, q); call cfill_z(-0.7d0, 0.3d0, t)
    call v1xypa(n, q, t, a2, r)
    call chk('v1xypa  r=q*t+a ('//"z"//')', &
             real(maxval(abs(r - (q*t + a2)))), 1.0E-12)

    call setup9c_z(x1, x2, x3, y1, y2, y3, w1, w2, w3)
    call pack_z(x1, x2, x3, xf); call pack_z(y1, y2, y3, yf)
    call pack_z(w1, w2, w3, wf)

    call v3dot(xf, yf, r)
    call chk('v3dot   r=x.y ('//"z"//')', &
             real(maxval(abs(r - rd_z(xf, yf)))), 1.0E-12)

    call v3had(xf, yf, wf)
    call chk('v3had   w=x.*y ('//"z"//')', &
             real(maxval(abs(wf - xf*yf))), 1.0E-12)

    call v3sqr(xf, r)
    call chk('v3sqr   r=x.x ('//"z"//')', &
             real(maxval(abs(r - rd_z(xf, xf)))), 1.0E-12)

    call cfill_z(0.3d0, -0.2d0, q)
    call v1norm(n, q, r)
    call chk('v1norm  r=sqrt(q.q) ('//"z"//')', &
             real(maxval(abs(r - sqrt(q*q)))), 1.0E-12)

    call v3cross(xf, yf, wf)
    call chk('v3cross w=x^y ('//"z"//')', &
             real(maxval(abs(wf - cr_z(xf, yf)))), 1.0E-12)

    call v3crossscal(xf, yf, al, wf)
    call chk('v3crossscal w=a.(x^y) ('//"z"//')', &
             real(maxval(abs(wf - al*cr_z(xf, yf)))), 1.0E-12)

    call v3crossdot(xf, yf, wf, r)
    call chk('v3crossdot r=(x^y).w ('//"z"//')', &
             real(maxval(abs(r - rd_z(cr_z(xf, yf), wf)))), 1.0E-12)

    call v3crosssqr(xf, yf, r)
    call chk('v3crosssqr r=(x^y).(x^y) ('//"z"//')', &
             real(maxval(abs(r - rd_z(cr_z(xf, yf), cr_z(xf, yf))))), 1.0E-12)

    call v3crossxy_crossxz(xf, yf, wf, uf, vf)
    call chk('v3crossxy_crossxz ('//"z"//')', real(max(&
             maxval(abs(uf - cr_z(xf, yf))), &
             maxval(abs(vf - cr_z(xf, wf))))), 1.0E-12)

    call v3crossxy_dotxz(xf, yf, wf, uf, r)
    call chk('v3crossxy_dotxz ('//"z"//')', real(max(&
             maxval(abs(uf - cr_z(xf, yf))), &
             maxval(abs(r - rd_z(xf, wf))))), 1.0E-12)

    call v3dotxy_dotxz(xf, yf, wf, r, q)
    call chk('v3dotxy_dotxz ('//"z"//')', real(max(&
             maxval(abs(r - rd_z(xf, yf))), &
             maxval(abs(q - rd_z(xf, wf))))), 1.0E-12)

    call v3crosscross(xf, yf, wf, rf)
    call chk('v3crosscross (z)', &
             real(maxval(abs(rf - cr_z(cr_z(xf, yf), wf)))), 1.0E-12)

    call v3norm_unit(xf, eps, r, uf)
    q = sqrt(sum(xf*xf, dim=2) + eps)
    call chk('v3norm_unit (z)', real(max(maxval(abs(r - q)), &
             maxval(abs(uf - xf/spread(q, 2, 3))))), 1.0E-12)

    call v3refl(xf, yf, 1.5d0, eps, rf)
    q = cmplx(1.5d0, 0.d0, kind=8)*rd_z(xf, yf)/(rd_z(yf, yf) + eps)
    call chk('v3refl (z)', real(maxval(abs(rf - &
             (xf - spread(q, 2, 3)*yf)))), 1.0E-12)

    call v3exb(xf, yf, 1.5d0, eps, rf)
    q = rd_z(yf, yf) + eps
    call chk('v3exb (z)', real(maxval(abs(rf - &
             cmplx(1.5d0, 0.d0, kind=8)*cr_z(xf, yf)/spread(q, 2, 3)))), 1.0E-12)

    call v3drag(xf, 1.5d0, eps, rf)
    q = sqrt(sum(xf*xf, dim=2) + eps)
    call chk('v3drag (z)', real(maxval(abs(rf - &
             cmplx(1.5d0, 0.d0, kind=8)*xf*spread(q, 2, 3)))), 1.0E-12)

    call cfill_z(0.3d0, -0.2d0, t)
    call v3mom_ke(t, xf, uf, q)
    call chk('v3mom_ke (z)', real(max(&
             maxval(abs(uf - spread(t, 2, 3)*xf)), &
             maxval(abs(q - h*t*rd_z(xf, xf))))), 1.0E-12)
  end subroutine test_cplx_z

end module v3blas_f90_cases

program v3blas_f90_test
  use v3blas_f90_cases
  use, intrinsic :: iso_c_binding, only: c_int
  implicit none
  integer(c_int), parameter :: n = 8
  character(len=32) :: arg

  if (command_argument_count() >= 1) then
     call get_command_argument(1, arg)
     select case (trim(arg))
     case ('s'); call test_real_s(n)
     case ('d'); call test_real_d(n)
     case ('c'); call test_cplx_c(n)
     case ('z'); call test_cplx_z(n)
     case default
        write(*, '(a)') 'usage: v3blas_f90_test [s|d|c|z]   (default: all)'
        stop 1
     end select
  else
     call test_real_s(n)
     call test_real_d(n)
     call test_cplx_c(n)
     call test_cplx_z(n)
  end if

  if (fails == 0) then
     write(*, '(a)') 'v3blas_f90_test: OK (0 failures)'
  else
     write(*, '(a, i0, a)') 'v3blas_f90_test: FAIL (', fails, ' failures)'
     stop 1
  end if
end program v3blas_f90_test
