
void diagonal(complex double *h1, complex double *eri, 
              int norb, int nelec, uint64_t *strs, 
              complex double *hdiag, uint64_t ndet);

void contract(complex double *h1, complex double *eri, 
              int norb, int nelec, uint64_t *strs, 
              complex double *civec, complex double *hdiag, 
              uint64_t ndet, complex double *ci1);

int n_excitations(uint64_t *str1, uint64_t *str2);
double compute_cre_des_sign(int a, int i, uint64_t *stria);
int *compute_occ_list(uint64_t *string, int norb, int nelec);
int *get_single_excitation(uint64_t *str1, uint64_t *str2);
int *get_double_excitation(uint64_t *str1, uint64_t *str2);

//int popcount(uint64_t bb);
//int trailz(uint64_t v);
extern int __builtin_popcountll (uint64_t x_0);
#define popcount(X) __builtin_popcountll((uint64_t) X)
extern int __builtin_ctzll (uint64_t x_0);
#define trailz(X) __builtin_ctzll((uint64_t) X)

