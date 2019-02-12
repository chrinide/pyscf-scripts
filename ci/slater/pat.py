#!/usr/bin/env python
def kernel(h1e, g2e, norb, nelec, ecore=0):

    h2e = absorb_h1e(h1e, g2e, norb, nelec, .5)

    na = cistring.num_strings(norb, nelec//2)
    ci0 = numpy.zeros((na,na))
    ci0[0,0] = 1

    # new code here
    neleca = nelecb = nelec//2
    nb = na
    ci_level = 3 # e.g. CISDT; make this kwarg if desired
    alpha_occs = fci.cistring._gen_occslst(range(norb), neleca)
    alpha_excitations = (alpha_occs >= neleca).sum(axis=1)
    beta_occs = fci.cistring._gen_occslst(range(norb), nelecb)
    beta_excitations = (beta_occs >= nelecb).sum(axis=1)
    a_idx, b_idx = numpy.array([
        [a,b] for a in alpha_excitations for b in beta_excitations if a+b > ci_level ]).T

    def hop(c):
        hc = contract_2e(h2e, c, norb, nelec)
        # new code here
        hc = hc.reshape(na,nb)
        hc[a_idx,b_idx] = 0
        return hc.reshape(-1)
    hdiag = make_hdiag(h1e, g2e, norb, nelec)
    # new code here
    hdiag = hdiag.reshape(na,nb)
    hdiag[a_idx,b_idx] = 0
    hdiag = hdiag.reshape(-1)
    precond = lambda x, e, *args: x/(hdiag-e+1e-4)
    e, c = lib.davidson(hop, ci0.reshape(-1), precond)
    return e+ecore
