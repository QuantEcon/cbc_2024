subroutine solow_fortran(k0, s, a, delta, alpha, n, kt)
 implicit none
 integer, parameter :: dp=kind(0.d0) 
 integer, intent(in) :: n
 real(dp), intent(in) :: k0, s, a, delta, alpha
 real(dp), intent(out) :: kt
 real(dp) :: k
 integer :: i
 k = k0
 do i = 1, n - 1                                                
  k = a * s * k**alpha + (1 - delta) * k
 end do
end subroutine solow_fortran 

