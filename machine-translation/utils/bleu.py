import math
import collections

SPTOK = set(['<sos>', '<pad>', '<eos>'])

def bleu_score(conds, refs, max_n=4, weights=[0.25]*4):
    conds = [[w for w in l if w not in SPTOK] for l in conds]
    refs = [[w for w in l if w not in SPTOK] for l in refs]
    cond_len, ref_len = 0, 0
    num_match, num_total = [0]*max_n, [0]*max_n
    for (cond, ref) in zip(conds, refs):
        cond_len += len(cond)
        ref_len += len(ref)
        for n in range(max_n):
            num_total[n] += len(cond) - n
            ref_counter = collections.defaultdict(int)
            for i in range(len(ref)-n):
                ref_counter[' '.join(ref[i:i+n+1])] += 1
            for i in range(len(cond)-n):
                if ref_counter[' '.join(cond[i:i+n+1])] > 0:
                    num_match[n] += 1
                    ref_counter[' '.join(cond[i:i+n+1])] -= 1
    score = math.exp(min(0, 1-ref_len/cond_len))
    for i in range(max_n):
        score *= math.pow(num_match[i]/num_total[i], weights[i])
    return score


if __name__ == '__main__':
    refs = [['the', 'cat', 'is', 'on', 'the', 'mat'], ['he', 'says']]
    conds = [['the', 'cat', 'sat', 'on', 'the', 'mat'], ['he', 'says']]
    print(bleu_score(conds, refs))
