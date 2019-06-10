function dfmp2(t2,eri,naux,nocc,nvir,eorb) bind(c)

  use iso_c_binding, only: c_double, c_int, c_size_t
  implicit none

  real(kind=c_double) :: dfmp2
  real(kind=c_double), intent(inout) :: t2(nocc,nvir,nocc,nvir)
  real(kind=c_double), intent(in) :: eri(naux,nocc,nvir)
  integer(kind=c_int), value, intent(in) :: naux, nocc, nvir
  real(kind=c_double), intent(in) :: eorb(nocc+nvir)

  real(kind=c_double) :: viajb, vibja, eps_i, eps_j, denom, tmp 
  integer(kind=c_int) :: i, j, loopij, loop1, ij, loop2, ab
  integer(kind=c_int) :: loopab, a, b, norb

  norb = nocc + nvir
  loop1 = nocc*nocc 
  loop2 = nvir*nvir 
  dfmp2 = 0.0

  do loopij = 1,loop1
    ij = mod(loopij-1,(nocc*nocc))
    i = ij/nocc + 1 
    j = mod(loopij-1,nocc) + 1
    eps_i = eorb(i)
    eps_j = eorb(j)
!$omp parallel default(none) &
!$omp private(loopab,ab,a,b,viajb,vibja,denom,tmp) &
!$omp shared(loop2,nvir,nocc,eri,i,j,eorb,eps_i,eps_j,t2) &
!$omp reduction(+:dfmp2)
!$omp do schedule(dynamic) 
    do loopab = 1,loop2
      ab = mod(loopab-1,(nvir*nvir))
      a = ab/nvir + 1 
      b = mod(loopab-1,nvir) + 1
      viajb = dot_product(eri(:,i,a),eri(:,j,b))
      vibja = dot_product(eri(:,i,b),eri(:,j,a))
      denom = eps_i + eps_j - eorb(a+nocc) - eorb(b+nocc) 
      tmp = (2.0*viajb - vibja)/denom 
      dfmp2 = dfmp2 + tmp*viajb
      t2(i,a,j,b) = tmp 
    end do
!$omp end do nowait 
!$omp end parallel
  end do

  return

end function dfmp2
