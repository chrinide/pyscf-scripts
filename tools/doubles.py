#!/usr/bin/env python

def doubleexcitations(nocc,nvir):
    doubles = []
    for occa in nocc:
        for occb in nocc:
            for virta in nvir:
                for virtb in nvir:
                    doubles.append((occa,occb,virta,virtb))
    return doubles
