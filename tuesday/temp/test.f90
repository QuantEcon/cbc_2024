
pure function capital(k0, s, a, delta, alpha, n)
 implicit none
 integer, parameter :: dp=kind(0.d0) 
 real(dp), intent(in) :: k0, s, a, delta, alpha
 real(dp) :: capital, r
 integer :: i
 integer, intent(in) :: n
 capital = k0
 do i = 1, n - 1                                                
  capital = a * s * capital**alpha + (1 - delta) * capital
 end do
 return
end function capital

program main
 implicit none
 integer, parameter :: dp=kind(0.d0)                          
 real(dp) :: start, finish, x, capital
 integer :: n
 real(dp) :: s=3.0_dp
 real(dp) :: a=1.0_dp
 real(dp) :: delta=0.1_dp
 real(dp) :: alpha=0.4_dp
 n = 1000000
 call cpu_time(start)
 x = capital(0.2_dp, s, a, delta, alpha, n)
 call cpu_time(finish)
 print *,'Last val = ', x
 print *,'Elapsed time = ', finish - start
end program main
