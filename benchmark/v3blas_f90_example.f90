! v3blas_f90_example.f90 - minimal tutorial for the v3blas Fortran
! front-end: generic names, plain arrays, nothing to free.
!
! A tiny particle loop: accelerate each point by its own drag term and
! print the kinetic-energy total of one step.
!
! Build (see CMakeLists.txt; v3blas.f90 comes from the install):
!   gfortran -O2 ../pkgs/openblas/include/openblas/v3blas.f90 \
!       v3blas_f90_example.f90 -L../pkgs/openblas/lib -lopenblas -lm \
!       -o v3blas_f90_example
program v3blas_f90_example
  use v3blas
  use, intrinsic :: iso_c_binding, only: c_int
  implicit none
  integer(c_int), parameter :: n = 4
  real(kind=8) :: x1(n), x2(n), x3(n)     ! velocities, one array per comp.
  real(kind=8) :: m1(n), m2(n), m3(n), ke(n)
  real(kind=8) :: mass(n)
  integer :: i

  do i = 1, n
     x1(i) = real(i, 8)          ! (1, 2, 3, 4)
     x2(i) = 0.0d0
     x3(i) = real(n - i, 8)
     mass(i) = 2.0d0
  end do

  ! m = rho*v, k = 0.5*rho*(v.v)   in one call
  call v3mom_ke(n, mass, x1, x2, x3, m1, m2, m3, ke)

  write(*, '(a,4f8.2)') 'momentum x: ', m1
  write(*, '(a,4f8.2)') 'kinetic:    ', ke
  write(*, '(a,f8.2,a)') 'total KE = ', sum(ke), ' (expected 44.0)'
end program v3blas_f90_example
