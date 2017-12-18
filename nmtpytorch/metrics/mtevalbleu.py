# -*- coding: utf-8 -*-
import re
import math

from collections import defaultdict

from .metric import Metric

# This is an exact reimplementation of mteval-v13a.pl
# It currently only works for single reference

LOG_2 = math.log(2)


def score_segment(tst_words, ref_words, ref_ngram_freqs, max_order):
    # Create initial lists
    match_cnt = [0 for i in range(max_order)]
    tst_cnt = [0 for i in range(max_order)]
    ref_cnt = [0 for i in range(max_order)]
    tst_info = [0 for i in range(max_order)]
    ref_info = [0 for i in range(max_order)]

    ref_ngrams_max = {}

    # Get the ngram counts for the test segment
    tst_ngrams = words_to_ngrams(tst_words, max_order)
    len_tst = len(tst_words)
    for i in range(max_order):
        tst_cnt[i] = (len_tst - i) if i < len_tst else 0

    ###########
    # Reference
    ###########
    ref_ngrams = words_to_ngrams(ref_words, max_order)
    len_ref = len(ref_words)
    for ngram_words, frq in ref_ngrams.items():
        # Counts of ngrams for this sentence
        ref_info[len(ngram_words) - 1] += ref_ngram_freqs[ngram_words]

        # Update the maximum count of this ngram
        # Shorter=>ref_ngrams_max[ngram_words] =
        #   max(ref_ngrams_max.get(ngram_words, -1), ref_ngrams[ngram_words])
        if ngram_words in ref_ngrams_max:
            ref_ngrams_max[ngram_words] = max(ref_ngrams_max[ngram_words], frq)
        else:
            ref_ngrams_max[ngram_words] = frq

    # Update reference ngram counts
    for i in range(max_order):
        ref_cnt[i] = (len_ref - i) if i < len_ref else 0

    for ngram_words, frq in tst_ngrams.items():
        if ngram_words in ref_ngrams_max:
            m = min(frq, ref_ngrams_max[ngram_words])
            len_ = len(ngram_words) - 1
            tst_info[len_] += ref_ngram_freqs[ngram_words] * m
            match_cnt[len_] += m

    return len_ref, match_cnt, tst_cnt, ref_cnt, tst_info, ref_info


def score_system(ref_segs, tst_segs, max_order):
    ref_ngram_freqs = compute_ngram_info(ref_segs, max_order)

    # 0-based indexing in contrast to perl version
    cum_match = [0 for i in range(max_order)]
    cum_tst_cnt = [0 for i in range(max_order)]
    cum_ref_cnt = [0 for i in range(max_order)]
    cum_tst_info = [0 for i in range(max_order)]
    cum_ref_info = [0 for i in range(max_order)]
    cum_ref_len = 0

    # Score each segment and keep statistics
    for tst, ref in zip(tst_segs, ref_segs):
        ref_len, match_cnt, tst_cnt, ref_cnt, tst_info, ref_info =\
            score_segment(tst, ref, ref_ngram_freqs, max_order)

        # Sum ref length
        cum_ref_len += ref_len

        for i in range(max_order):
            cum_match[i] += match_cnt[i]
            cum_tst_cnt[i] += tst_cnt[i]
            cum_ref_cnt[i] += ref_cnt[i]
            cum_tst_info[i] += tst_info[i]
            cum_ref_info[i] += ref_info[i]

    # Compute length score
    exp_len_score = math.exp(min(0, 1 - cum_ref_len / cum_tst_cnt[0])) \
        if cum_tst_cnt[0] > 0 else 0

    # For further length ratio computation
    tst_vs_ref_ratio = (cum_tst_cnt[0], cum_ref_len, math.log(exp_len_score))

    return bleu_score(cum_ref_len, cum_match, cum_tst_cnt,
                      exp_len_score, max_order), tst_vs_ref_ratio


def read_sentences(filename, tokenizer, is_cased):
    """Read simple plain text file or from list."""
    if isinstance(filename, list):
        data = filename
    elif isinstance(filename, str):
        data = open(filename)

    sents = []
    for line in data:
        sents.append(tokenizer(line.strip(), is_cased))
    return sents


def words_to_ngrams(words, max_order):
    """Convert a sequence of words to an ngram count dict."""
    d = defaultdict(int)

    # Iterate over word indices as start pointers
    for i in range(len(words)):
        # Sliding windows
        for j in range(min(max_order, len(words) - i)):
            # Increment counter, keep keys as tuples
            d[tuple(words[i: i + j + 1])] += 1

    return d


def compute_ngram_info(ref_segs, max_order):
    tot_words = 0

    # Segment-wise frequencies
    ngram_count = defaultdict(int)

    ngram_info = {}

    for words in ref_segs:
        tot_words += len(words)
        # Get frequencies and add them to ngramcpunt
        for key, value in words_to_ngrams(words, max_order).items():
            ngram_count[key] += value

    for ngram_words, freq in ngram_count.items():
        if len(ngram_words) == 1:
            # ngram is unigram => corpus frequency
            denom = tot_words
        else:
            # n-gram is n-gram => n-gram frequency
            denom = ngram_count[ngram_words[:-1]]

        ngram_info[ngram_words] = -math.log(freq / denom) / LOG_2
    return ngram_info


def bleu_score(ref_len, matching_ngrams, tst_ngrams, exp_len_score, max_order):
    score = 0
    iscore = 0
    smooth = 1

    ind_scores = []
    cum_scores = []

    for i in range(max_order):
        if tst_ngrams[i] == 0:
            iscore = 0
        elif matching_ngrams[i] == 0:
            smooth *= 2
            iscore = math.log(1 / (smooth * tst_ngrams[i]))
        else:
            iscore = math.log(matching_ngrams[i] / tst_ngrams[i])

        ind_scores.append(math.exp(iscore))
        score += iscore
        cum_scores.append(math.exp(score / (i + 1)) * exp_len_score)

    return ind_scores, cum_scores


def tokenizer(s, is_cased):
    s = s.strip()

    # language-independent part:
    if '<skipped>' in s:
        s = re.sub('<skipped>', '', s)

    # language-dependent part (assuming Western languages):
    s = " %s " % s
    if not is_cased:
        s = s.lower()

    # tokenize punctuation
    s = re.sub('([\{-\~\[-\` -\&\(-\+\:-\@\/])', ' \\1 ', s)

    # tokenize period and comma unless preceded by a digit
    s = re.sub('([^0-9])([\.,])', '\\1 \\2 ', s)

    # tokenize period and comma unless followed by a digit
    s = re.sub('([\.,])([^0-9])', ' \\1 \\2', s)

    # tokenize dash when preceded by a digit
    if '-' in s:
        s = re.sub('([0-9])(-)', '\\1 \\2 ', s)

    # Strip multiple spaces
    # Strip trailing and leading spaces
    return re.sub('\s+', ' ', s).strip().split()


class MTEvalV13aBLEUScorer(object):
    def compute(self, refs, hyps):
        # Make reference files a list
        refs = [refs] if isinstance(refs, str) else refs

        assert len(refs) > 1, \
            "mteval-v13a does not currently support multi refs."

        # Take the first one
        ref_file = refs[0]

        # Read (detokenized) files and tokenize them
        ref_segs = read_sentences(ref_file, tokenizer, True)
        tst_segs = read_sentences(hyps, tokenizer, True)

        assert(len(ref_segs) == len(tst_segs))

        (ind_scores, cum_scores), ratios = score_system(ref_segs, tst_segs, 4)

        float_score = cum_scores[3] * 100

        bleu_str = "%.2f, %s (ratio=%.3f, hyp_len=%d, ref_len=%d)" % (
            float_score,
            "/".join([("%.1f" % (s * 100)) for s in ind_scores[:4]]),
            (ratios[0] / ratios[1]), ratios[0], ratios[1])

        return Metric('BLEUv13a', float_score, bleu_str)
