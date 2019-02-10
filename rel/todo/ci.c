#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <stdint.h>
#include "ci.h"

// Computes <i|H|i>
void diagonal(double complex *h1, double complex *diagj, 
              double complex *diagk, int norb, int nelec, 
              uint64_t *strs, uint64_t ndet, double complex *hdiag){
}

// Computes C'=H*C
void contract(double complex *h1, double complex *eri, 
              int norb, int nelec, uint64_t *strs, 
              double complex *civec, double complex *hdiag, 
              uint64_t ndet, double complex *ci1){

  int *ts = malloc(sizeof(int)*ndet); 
  #pragma omp parallel default(none) shared(strs, ndet, ts)
  {
    size_t ip;
    uint64_t *str1 = strs;
    ts[0] = 0;
    #pragma omp for schedule(static)
    for (ip=1; ip<ndet; ++ip){
      uint64_t *stri = strs + ip;
      ts[ip] = n_excitations(stri, str1);
    }
  }

  //#pragma omp parallel default(none) shared(strs, ndet, ts, h1, eri, norb, nelec, civec, hdiag, ci1)
  //{
  size_t ip, jp;
  //#pragma omp for schedule(dynamic)
  for (ip=0; ip<ndet; ++ip){
    for (jp=0; jp<ip; ++jp){
      if (abs(ts[ip] - ts[jp]) < 3){
        uint64_t *stri = strs + ip;
        uint64_t *strj = strs + jp;
        int n_excit = n_excitations(stri, strj);
        // Single excitation
        if (n_excit == 1){
          int p;
          int *ia = get_single_excitation(stri, strj);
          int i = ia[0];
          int a = ia[1];
          double complex fai = h1[a*norb + i];
          int *occs = compute_occ_list(stri, norb, nelec);
          for (p=0; p<nelec; ++p){
            int k = occs[p];
            size_t kkai = k*norb*norb*norb + k*norb*norb + a*norb + i;
            size_t kiak = k*norb*norb*norb + i*norb*norb + a*norb + k;
            fai += eri[kkai] - eri[kiak];
          }
          if (fabs(fai) > 1.0e-14){
            double sign = compute_cre_des_sign(a, i, stri);
            fai *= sign;
            //#pragma omp critical
            ci1[ip] += fai*civec[jp];
            ci1[jp] += conj(fai)*civec[ip];
          }
          free(occs);
        }
        // Double excitation
        else if (n_excit == 2){
	        int *ijab = get_double_excitation(stri, strj);
          int i = ijab[0]; int j = ijab[1]; int a = ijab[2]; int b = ijab[3];
          double complex v;
          double sign;
          size_t ajbi = a*norb*norb*norb + j*norb*norb + b*norb + i;
          size_t aibj = a*norb*norb*norb + i*norb*norb + b*norb + j;
          if (a > j || i > b){
            v = eri[ajbi] - eri[aibj];
            sign = compute_cre_des_sign(b, i, stri);
            sign *= compute_cre_des_sign(a, j, stri);
          } 
          else {
            v = eri[aibj] - eri[ajbi];
            sign = compute_cre_des_sign(b, j, stri);
            sign *= compute_cre_des_sign(a, i, stri);
          }
          if (fabs(v) > 1.0e-14){
            v *= sign;
            //#pragma omp critical
            ci1[ip] += v*civec[jp];
            ci1[jp] += conj(v)*civec[ip];
          }
          free(ijab);
        }
        else {
          continue;
        }
      }
    }
    //#pragma omp critical
    ci1[ip] += hdiag[ip]*civec[ip]; // Diagonal term 
  }
  //}
  free(ts);
}

// Compute a list of occupied orbitals for a given string
int *compute_occ_list(uint64_t *string, int norb, int nelec){
  int *occ = malloc(sizeof(int)*nelec);
  int i, occ_ind = 0;
  for (i=0; i<norb; i++){
    int i_occ = (string[0] >> i) & 1;
    if (i_occ){
      occ[occ_ind] = i;
      occ_ind++;
    }
  }
  return occ;
}

// Compare two strings and compute excitation level
inline int n_excitations(uint64_t *str1, uint64_t *str2){
  int d = 0;
  d = popcount(str1[0] ^ str2[0]);
  return d/2;
}

inline double compute_cre_des_sign(int p, int q, uint64_t *string0){
  uint64_t mask;
  if (p > q){
    mask = (1ULL << p) - (1ULL << (q+1));
  } 
  else {
    mask = (1ULL << q) - (1ULL << (p+1));
  }
  if (popcount(string0[0] & mask) % 2){
    return -1.0;
  } 
  else {
    return 1.0;
  }
}

// Compute orbital indices for a single excitation 
int *get_single_excitation(uint64_t *str1, uint64_t *str2){
  int *ia = malloc(sizeof(int)*2);
  uint64_t str_tmp = str1[0] ^ str2[0];
  uint64_t str_particle = str_tmp & str2[0];
  uint64_t str_hole = str_tmp & str1[0];
  if (popcount(str_particle) == 1){
    ia[1] = trailz(str_particle);
  }
  if (popcount(str_hole) == 1){
    ia[0] = trailz(str_hole);
  }
  return ia;
}

// Compute orbital indices for a double excitation 
int *get_double_excitation(uint64_t *str1, uint64_t *str2){
  int *ijab = malloc(sizeof(int)*4);
  int particle_ind = 2;
  int hole_ind = 0;
  uint64_t str_tmp = str1[0] ^ str2[0];
  uint64_t str_particle = str_tmp & str2[0];
  uint64_t str_hole = str_tmp & str1[0];
  int n_particle = popcount(str_particle);
  int n_hole = popcount(str_hole);
  if (n_particle == 1){
    ijab[particle_ind] = trailz(str_particle);
    particle_ind++;
  }
  else if (n_particle == 2){
    int a = trailz(str_particle);
    ijab[2] = a;
    str_particle &= ~(1ULL << a);
    int b = trailz(str_particle);
    ijab[3] = b;
  }
  if (n_hole == 1){
    ijab[hole_ind] = trailz(str_hole);
    hole_ind++;
  }
  else if (n_hole == 2){
    int i = trailz(str_hole);
    ijab[0] = i;
    str_hole &= ~(1ULL << i);
    int j = trailz(str_hole);
    ijab[1] = j;
  }
  return ijab;
}

// Compute number of trailing zeros in a bit string
int trailz(uint64_t v){
  int c = 64;
  // Trick to unset all bits but the first one
  v &= -(int64_t) v;
  if (v) c--;
  if (v & 0x00000000ffffffff) c -= 32;
  if (v & 0x0000ffff0000ffff) c -= 16;
  if (v & 0x00ff00ff00ff00ff) c -= 8;
  if (v & 0x0f0f0f0f0f0f0f0f) c -= 4;
  if (v & 0x3333333333333333) c -= 2;
  if (v & 0x5555555555555555) c -= 1;
  return c;
}

int popcount(uint64_t x){
  const uint64_t m1  = 0x5555555555555555; //binary: 0101...
  const uint64_t m2  = 0x3333333333333333; //binary: 00110011..
  const uint64_t m4  = 0x0f0f0f0f0f0f0f0f; //binary:  4 zeros,  4 ones ...
  const uint64_t m8  = 0x00ff00ff00ff00ff; //binary:  8 zeros,  8 ones ...
  const uint64_t m16 = 0x0000ffff0000ffff; //binary: 16 zeros, 16 ones ...
  const uint64_t m32 = 0x00000000ffffffff; //binary: 32 zeros, 32 ones
  x = (x & m1 ) + ((x >>  1) & m1 ); //put count of each  2 bits into those  2 bits 
  x = (x & m2 ) + ((x >>  2) & m2 ); //put count of each  4 bits into those  4 bits 
  x = (x & m4 ) + ((x >>  4) & m4 ); //put count of each  8 bits into those  8 bits 
  x = (x & m8 ) + ((x >>  8) & m8 ); //put count of each 16 bits into those 16 bits 
  x = (x & m16) + ((x >> 16) & m16); //put count of each 32 bits into those 32 bits 
  x = (x & m32) + ((x >> 32) & m32); //put count of each 64 bits into those 64 bits 
  return x;
}

