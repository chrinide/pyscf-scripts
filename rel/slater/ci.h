
void diagonal(double complex *h1, double complex *diagj, 
              double complex *diagk, int norb, int nelec, 
              uint64_t *strs, uint64_t ndet, double complex *hdiag);

void contract(double complex *h1, double complex *eri, 
              int norb, int nelec, uint64_t *strs, 
              double complex *civec, double complex *hdiag, 
              uint64_t ndet, double complex *ci1);

void rdm1(int norb, int nelec, uint64_t *strs, 
          double complex *civec, uint64_t ndet, 
          double complex *rdm1);
void rdm12(int norb, int nelec, uint64_t *strs, 
           double complex *civec, uint64_t ndet, 
           double complex *rdm1, double complex *rdm2);

int n_excitations(uint64_t *str1, uint64_t *str2);
double compute_cre_des_sign(int a, int i, uint64_t *stria);
int *compute_occ_list(uint64_t *string, int norb, int nelec);
int *get_single_excitation(uint64_t *str1, uint64_t *str2);
int *get_double_excitation(uint64_t *str1, uint64_t *str2);
int *get_triple_excitation(uint64_t *str1, uint64_t *str2);

//int popcount(uint64_t bb);
//int trailz(uint64_t v);

extern int __builtin_popcountll (uint64_t x_0);
#define popcount(X) __builtin_popcountll((uint64_t) X)
extern int __builtin_ctzll (uint64_t x_0);
#define trailz(X) __builtin_ctzll((uint64_t) X)

