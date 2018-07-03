#!/usr/bin/env python
nocc = 3
#pairs = nocc*(nocc+1)/2
pairs = nocc**2
print pairs
for ij in range(pairs):
    tmp = ij % (nocc*nocc)
    i = tmp//nocc
    j = ij % nocc
    print i,j

