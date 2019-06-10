#include <stdlib.h>
#include <stdio.h>
#include "mp2.h"

double mp2(double *t2, const double *eri, 
           const int norb, const int nocc, 
           const double *eorb) {

  const int nvir = norb - nocc;
  double e_mp2 = 0.0;
  double ev[nvir], eo[nocc];
  const int loop1 = nocc*nocc;
  const int loop2 = nvir*nvir;
  int i,j,a,b, loopij, loopab;

  for (i=0; i<nocc; i++){
    eo[i] = eorb[i];
  }
  for (i=0; i<nvir; i++){
    ev[i] = eorb[nocc+i];
  }

  for (loopij=0; loopij<loop1; loopij++){
    int ij = loopij%(nocc*nocc);
    i = ij/nocc;
    j = loopij%nocc;
    size_t ii = i*(nvir*nocc*nvir);
    int jj = j*nvir;
#pragma omp parallel default(none) reduction(+:e_mp2) \
    shared(eo,ev,eri,t2,jj,ii,i,j)
{
#pragma omp for nowait schedule(dynamic)
    for (loopab=0; loopab<loop2; loopab++){
      int ab = loopab%(nvir*nvir);
      int a = ab/nvir;
      int b = loopab%nvir;
      int aa = a*(nvir*nocc);
      int bbt = b*(nvir*nocc);
      size_t offset = ii + aa + jj + b;
      size_t offsett = ii + bbt + jj + a;
      double denom = eo[i] + eo[j] - ev[a] - ev[b];
      double tmp = (2.0*eri[offset] - eri[offsett])/denom;
      e_mp2 += tmp*eri[offset];
      t2[offset] = tmp;
    }
}
  }

  return e_mp2;

}
