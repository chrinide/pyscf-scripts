#!/usr/bin/env python

ci_level = 3 # e.g. CISDT; make this kwarg if desired
alpha_occs = fci.cistring._gen_occslst(range(norb), neleca)
alpha_excitations = (alpha_occs >= neleca).sum(axis=1)
beta_occs = fci.cistring._gen_occslst(range(norb), nelecb)
beta_excitations = (beta_occs >= nelecb).sum(axis=1)
a_idx, b_idx = numpy.array([
    [a,b] for a in alpha_excitations for b in beta_excitations if a+b > ci_level ]).T
hc[a_idx,b_idx] = 0

