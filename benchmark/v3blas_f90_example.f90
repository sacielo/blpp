! v3blas_f90_example.f90 - minimal tutorial for the v3blas Fortran
! front-end: generic names, one field array u(n, 3) per 3-vector
! (column c = component c), nothing to free.
!
! A tiny particle loop: build velocities, compute momentum and kinetic
! energy in one call, print a component (a column of the field).
!
! Build (see CMakeLists.txt; v3blas.f90 comes from the install):
!   gfortran -O2 ../pkgs/openblas/include/openblas/v3blas.f90 \
!       v3blas_f90_example.f90 -L../pkgs/openblas/lib -lopenblas -lm \
!       -o v3blas_f90_example
program v3blas_f90_example
  use v3blas
  use, intrinsic :: iso_c_binding, only: c_int
  implicit none
  integer, parameter :: n = 4
  real(kind=8) :: v(n, 3)               ! velocities: one row per point
  real(kind=8) :: mom(n, 3), ke(n), mass(n)
  integer :: i

  do i = 1, n
     v(i, :) = [real(i, 8), 0.0d0, real(n - i, 8)]
     mass(i) = 2.0d0
  end do

  ! m = rho*v, k = 0.5*rho*(v.v)   in one call; n is implicit
  call v3mom_ke(mass, v, mom, ke)

  write(*, '(a,4f8.2)') 'momentum x: ', mom(:, 1)
  write(*, '(a,4f8.2)') 'kinetic:    ', ke
  write(*, '(a,f8.2,a)') 'total KE = ', sum(ke), ' (expected 44.0)'
end program v3blas_f90_example
