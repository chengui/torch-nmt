import math
import collections


def bleu_score(conds, refs, max_n=4, weights=[0.25]*4):
    cond_len, ref_len = 0, 0
    num_match, num_total = [0]*max_n, [0]*max_n
    for (cond, ref) in zip(conds, refs):
        cond_len += len(cond)
        ref_len += len(ref)
        for n in range(max_n):
            if len(ref) <= n:
                num_match[n] = -1
                break
            if len(cond) <= n:
                num_total[n] = 1
                continue
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
        if num_match[i] < 0:
            break
        score *= math.pow(num_match[i]/num_total[i], weights[i])
    return score


if __name__ == '__main__':
    refs = [['the', 'cat', 'is', 'on', 'the', 'mat'], ['he', 'says']]
    conds = [['the', 'cat', 'sat', 'on', 'the', 'mat'], ['he', 'says']]
    print(bleu_score(conds, refs))
