import math
import collections


def ngram_counter(seq, n):
    counts = collections.Counter()
    for i in range(n):
        for k in range(0, len(seq) - i):
            ngram = tuple(seq[k:k+i+1])
            counts[ngram] += 1
    return counts

def bleu_score(cands, refss, max_n=4, weights=[0.25]*4, smooth=False):
    cand_len = 0
    refs_len = 0
    total_counts = [0] * max_n
    clipped_counts = [0] * max_n
    for (cand, refs) in zip(cands, refss):
        cand_len += len(cand)
        refs_len += min([len(ref) for ref in refs],
                        key=lambda i: abs(len(cand) - i))

        merged = collections.Counter()
        for ref in refs:
            merged |= ngram_counter(ref, max_n)
        cand_counter = ngram_counter(cand, max_n)
        clipped = cand_counter & merged

        for ngram in clipped:
            clipped_counts[len(ngram)-1] += clipped[ngram]
        for n in range(max_n):
            total_counts[n] += max(0, len(cand)-n)

    pn = [0] * max_n
    for n in range(max_n):
        if smooth:
            pn[n] = (clipped_counts[n] + 1.) / (total_counts[n] + 1.)
        elif total_counts[n] > 0:
            pn[n] = float(clipped_counts[n]) / total_counts[n]
        else:
            pn[n] = 0.0

    if min(pn) == 0:
        return 0.0, tuple(pn)

    log_pn = sum(w * math.log(p) for (w, p) in zip(weights, pn))
    log_bp = min(0, 1 - cand_len/refs_len)

    return math.exp(log_bp + log_pn), tuple(pn)


if __name__ == '__main__':
    refss = [[['the', 'cat', 'is', 'on', 'the', 'mat']], [['he', 'says']]]
    conds = [['the', 'cat', 'sat', 'on', 'the', 'mat'], ['he', 'says']]
    #print(ngram_counter(conds[0], 4))
    print(bleu_score(conds, refss, smooth=True))
