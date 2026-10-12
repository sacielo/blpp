! v3blas_f90_bench.f90 - timing for the v3blas Fortran front-end.
! Four representative ops per real kind (the full generic matrix is
! v3blas_f90_test's job; timing only, nothing is verified here).  The
! budget loop mirrors the C++ benches: run until 0.5 s or 2000 calls.
! One flat array per vector component: a strided slice such as x(1,:)
! would be a hidden temporary copy and would time the compiler, not us.
!
! Build (see CMakeLists.txt; v3blas.f90 comes from the install):
!   gfortran -O2 ../pkgs/openblas/include/openblas/v3blas.f90 \
!       v3blas_f90_bench.f90 -L../pkgs/openblas/lib -lopenblas -lm \
!       -o v3blas_f90_bench
program v3blas_f90_bench
  use v3blas
  use, intrinsic :: iso_c_binding, only: c_int
  implicit none
  integer, parameter :: n = 4096, nmax = 2000
  integer(c_int) :: nn = n
  real(kind=4), parameter :: a4 = 1.5, e4 = 0.25
  real(kind=8), parameter :: a8 = 1.5d0, e8 = 0.25d0
  real(kind=4) :: x1(n), x2(n), x3(n), y1(n), y2(n), y3(n)
  real(kind=4) :: r1(n), r2(n), r3(n), r4(n)
  real(kind=8) :: dx1(n), dx2(n), dx3(n), dy1(n), dy2(n), dy3(n)
  real(kind=8) :: dr1(n), dr2(n), dr3(n), dr4(n)
  integer :: i, sel

  do i = 1, n
     x1(i) = sin(real(i));        x2(i) = sin(real(i) + 1.0)
     x3(i) = sin(real(i) + 2.0);  y1(i) = cos(real(i))
     y2(i) = cos(real(i) + 1.0);  y3(i) = cos(real(i) + 2.0)
     dx1(i) = dsin(dble(i));       dx2(i) = dsin(dble(i) + 1.d0)
     dx3(i) = dsin(dble(i) + 2.d0); dy1(i) = dcos(dble(i))
     dy2(i) = dcos(dble(i) + 1.d0); dy3(i) = dcos(dble(i) + 2.d0)
  end do

  write(*, '(a)') 'v3blas_f90_bench: timing only (correctness: v3blas_f90_test)'
  write(*, '(a,i0,a)') 'n = ', n, '   budget: 0.5 s or 2000 calls per kernel'
  write(*, '(a11,a12,a11,1x,a)') 'kernel', 'ms/run', 'GFLOP/s', 'equation'

  do sel = 1, 4
     call run_s(sel)
     call run_d(sel)
  end do

contains

  ! sel: 1 v3dot(5)  2 v3cross(9)  3 v3drag(27)  4 v3mom_ke(10)
  subroutine run_s(sel)
    integer, intent(in) :: sel
    integer :: k, c0, c1, cr
    real :: el
    character(len=9) :: nm
    character(len=20) :: eq
    real, parameter :: fl(4) = [5.0, 9.0, 27.0, 10.0]
    call describe(sel, nm, eq)
    call system_clock(c0, cr)
    k = 0
    do
       select case (sel)
       case (1)
          call v3dot(nn, x1, x2, x3, y1, y2, y3, r1)
       case (2)
          call v3cross(nn, x1, x2, x3, y1, y2, y3, r1, r2, r3)
       case (3)
          call v3drag(nn, x1, x2, x3, a4, e4, r1, r2, r3)
       case (4)
          call v3mom_ke(nn, y1, x1, x2, x3, r1, r2, r3, r4)
       end select
       k = k + 1
       call system_clock(c1)
       if (k >= nmax .or. real(c1 - c0) / cr > 0.5) exit
    end do
    el = real(c1 - c0) / cr
    write(*, '(a9,a2,f11.4,f10.3,1x,a)') trim(nm), '_s', el / k * 1000.0, &
      fl(sel) * n * k / el / 1.0e9, eq
  end subroutine run_s

  subroutine run_d(sel)
    integer, intent(in) :: sel
    integer :: k, c0, c1, cr
    real :: el
    character(len=9) :: nm
    character(len=20) :: eq
    real, parameter :: fl(4) = [5.0, 9.0, 27.0, 10.0]
    call describe(sel, nm, eq)
    call system_clock(c0, cr)
    k = 0
    do
       select case (sel)
       case (1)
          call v3dot(nn, dx1, dx2, dx3, dy1, dy2, dy3, dr1)
       case (2)
          call v3cross(nn, dx1, dx2, dx3, dy1, dy2, dy3, dr1, dr2, dr3)
       case (3)
          call v3drag(nn, dx1, dx2, dx3, a8, e8, dr1, dr2, dr3)
       case (4)
          call v3mom_ke(nn, dy1, dx1, dx2, dx3, dr1, dr2, dr3, dr4)
       end select
       k = k + 1
       call system_clock(c1)
       if (k >= nmax .or. real(c1 - c0) / cr > 0.5) exit
    end do
    el = real(c1 - c0) / cr
    write(*, '(a9,a2,f11.4,f10.3,1x,a)') trim(nm), '_d', el / k * 1000.0, &
      fl(sel) * n * k / el / 1.0e9, eq
  end subroutine run_d

  subroutine describe(sel, nm, eq)
    integer, intent(in) :: sel
    character(len=*), intent(out) :: nm, eq
    select case (sel)
    case (1); nm = 'v3dot';    eq = 'r=x1y1+x2y2+x3y3'
    case (2); nm = 'v3cross';  eq = 'w=x^y'
    case (3); nm = 'v3drag';   eq = 'r=a*x*sqrt(x.x+eps)'
    case (4); nm = 'v3mom_ke'; eq = 'm=s*x, k=0.5*s*(x.x)'
    end select
  end subroutine describe

end program v3blas_f90_bench
