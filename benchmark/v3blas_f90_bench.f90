! v3blas_f90_bench.f90 - timing for the v3blas Fortran front-end.
! Four representative ops per real kind (the full generic matrix is
! v3blas_f90_test's job; timing only, nothing is verified here).  The
! budget loop mirrors the C++ benches: run until 0.005 s or 2000 calls.
! Fields are u(n, 3): the columns are the contiguous component arrays,
! so each call maps straight onto the SoA layout the kernels read.
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
  real(kind=4), parameter :: a4 = 1.5, e4 = 0.25
  real(kind=8), parameter :: a8 = 1.5d0, e8 = 0.25d0
  real(kind=4) :: xf(n, 3), yf(n, 3), rf(n, 3), s4(n), k4(n)
  real(kind=8) :: dxf(n, 3), dyf(n, 3), drf(n, 3), s8(n), k8(n)
  integer :: i, sel

  do i = 1, n
     xf(i, 1) = sin(real(i));       xf(i, 2) = sin(real(i) + 1.0)
     xf(i, 3) = sin(real(i) + 2.0)
     yf(i, 1) = cos(real(i));       yf(i, 2) = cos(real(i) + 1.0)
     yf(i, 3) = cos(real(i) + 2.0)
     dxf(i, 1) = dsin(dble(i));     dxf(i, 2) = dsin(dble(i) + 1.d0)
     dxf(i, 3) = dsin(dble(i) + 2.d0)
     dyf(i, 1) = dcos(dble(i));     dyf(i, 2) = dcos(dble(i) + 1.d0)
     dyf(i, 3) = dcos(dble(i) + 2.d0)
  end do
  s4 = 0.5; s8 = 0.5d0

  write(*, '(a)') 'v3blas_f90_bench: timing only (correctness: v3blas_f90_test)'
  write(*, '(a,i0,a)') 'n = ', n, '   budget: 0.005 s or 2000 calls per kernel'
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
          call v3dot(xf, yf, s4)
       case (2)
          call v3cross(xf, yf, rf)
       case (3)
          call v3drag(xf, a4, e4, rf)
       case (4)
          call v3mom_ke(s4, xf, rf, k4)
       end select
       k = k + 1
       call system_clock(c1)
       if (k >= nmax .or. real(c1 - c0) / cr > 0.005) exit
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
          call v3dot(dxf, dyf, s8)
       case (2)
          call v3cross(dxf, dyf, drf)
       case (3)
          call v3drag(dxf, a8, e8, drf)
       case (4)
          call v3mom_ke(s8, dxf, drf, k8)
       end select
       k = k + 1
       call system_clock(c1)
       if (k >= nmax .or. real(c1 - c0) / cr > 0.005) exit
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
