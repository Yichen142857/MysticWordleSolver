"""Compiled inner loops for the cipher-search baselines.

Everything works in the *decoder* view: d[cipher_letter] = plaintext_letter,
i.e. d = c^-1. Because score(g, c(t)) == score(c^-1(g), t), a decoder turns
every past guess into a decoded guess that is scored directly against
plaintext candidate words, so targets can be filtered without enumerating them
jointly with the cipher.

Letters are uint8 0..25; word rows are padded with PAD to MAXLEN.
If numba is missing, the same code runs as plain Python (correct, but slow).
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - fallback path
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda f: f

MAXLEN = 10
PAD = 255
FB_BASE = 128  # per-target feedback code: green * 11 + total < 128


# -- encoding (plain Python / numpy) -------------------------------------------

def encode_words(words):
    """-> pos (W x MAXLEN uint8, PAD-padded), lens (W,), cnt (W x 26 uint8)."""
    W = len(words)
    pos = np.full((W, MAXLEN), PAD, dtype=np.uint8)
    lens = np.zeros(W, dtype=np.int64)
    cnt = np.zeros((W, 26), dtype=np.uint8)
    for i, w in enumerate(words):
        b = np.frombuffer(w.encode(), dtype=np.uint8) - 97
        pos[i, : len(w)] = b
        lens[i] = len(w)
        np.add.at(cnt[i], b, 1)
    return pos, lens, cnt


def guess_rows(word):
    """-> (pos row, length, distinct letters row, their counts row, n distinct)."""
    pos = np.full(MAXLEN, PAD, dtype=np.uint8)
    b = np.frombuffer(word.encode(), dtype=np.uint8) - 97
    pos[: len(word)] = b
    letters, counts = np.unique(b, return_counts=True)
    dl = np.full(MAXLEN, PAD, dtype=np.uint8)
    dc = np.zeros(MAXLEN, dtype=np.uint8)
    dl[: len(letters)] = letters
    dc[: len(letters)] = counts
    return pos, len(word), dl, dc, len(letters)


# -- constraint evaluation ------------------------------------------------------

@njit(cache=True)
def _decode_guesses(d, g_pos, g_len, g_dl, g_nd, K):
    dg = np.full((K, MAXLEN), PAD, dtype=np.uint8)
    ddl = np.full((K, MAXLEN), PAD, dtype=np.uint8)
    for k in range(K):
        for j in range(g_len[k]):
            dg[k, j] = d[g_pos[k, j]]
        for j in range(g_nd[k]):
            ddl[k, j] = d[g_dl[k, j]]
    return dg, ddl


@njit(cache=True)
def _word_violation(w, dg, ddl, g_len, g_dc, g_nd, K, fb, i, w_pos, w_len, w_cnt, bound):
    """Sum over guesses of |green - fg| + |total - ft| for word w and target i.
    Stops early once the sum reaches `bound` (the result is then >= bound)."""
    acc = 0
    for k in range(K):
        fg = fb[k, i, 0]
        if fg < 0:
            continue
        ft = fg + fb[k, i, 1]
        m = g_len[k] if g_len[k] < w_len[w] else w_len[w]
        gr = 0
        for j in range(m):
            if dg[k, j] == w_pos[w, j]:
                gr += 1
        tot = 0
        for j in range(g_nd[k]):
            c = w_cnt[w, ddl[k, j]]
            gc = g_dc[k, j]
            tot += c if c < gc else gc
        acc += abs(gr - fg) + abs(tot - ft)
        if acc >= bound:
            return acc
    return acc


@njit(cache=True)
def target_stats(d, g_pos, g_len, g_dl, g_dc, g_nd, K, fb, w_pos, w_len, w_cnt, lo, hi, v_out, nz_out):
    """For decoder d, per target i: v = min violation over candidate words
    [lo[i], hi[i]), nz = number of candidates with zero violation."""
    dg, ddl = _decode_guesses(d, g_pos, g_len, g_dl, g_nd, K)
    for i in range(lo.shape[0]):
        best = 1 << 30
        nz = 0
        for w in range(lo[i], hi[i]):
            bound = 1 if best == 0 else best
            acc = _word_violation(w, dg, ddl, g_len, g_dc, g_nd, K, fb, i, w_pos, w_len, w_cnt, bound)
            if acc == 0:
                nz += 1
                best = 0
            elif acc < best:
                best = acc
        v_out[i] = best
        nz_out[i] = nz


@njit(cache=True)
def log_target(v, nz, beta):
    """Tempered log posterior of a decoder: sum_i log nz_i if target i has a
    consistent word, else -beta * v_i. For consistent decoders this is the exact
    log posterior up to a constant (targets summed out, uniform within length)."""
    s = 0.0
    for i in range(v.shape[0]):
        if v[i] == 0:
            s += np.log(nz[i])
        else:
            s -= beta * v[i]
    return s


@njit(cache=True)
def best_words(d, g_pos, g_len, g_dl, g_dc, g_nd, K, fb, w_pos, w_len, w_cnt, lo, hi, i, v):
    """Indices of target i's candidates whose violation equals v (its minimum)."""
    dg, ddl = _decode_guesses(d, g_pos, g_len, g_dl, g_nd, K)
    out = np.empty(hi[i] - lo[i], dtype=np.int64)
    n = 0
    for w in range(lo[i], hi[i]):
        if _word_violation(w, dg, ddl, g_len, g_dc, g_nd, K, fb, i, w_pos, w_len, w_cnt, v + 1) == v:
            out[n] = w
            n += 1
    return out[:n]


@njit(cache=True)
def seed_rng(seed):
    np.random.seed(seed)


@njit(cache=True)
def _viol_direct(d, w, i, g_pos, g_len, g_dl, g_dc, g_nd, K, fb, w_pos, w_len, w_cnt):
    """Violation of word w for target i under decoder d (decoding on the fly)."""
    acc = 0
    for k in range(K):
        fg = fb[k, i, 0]
        if fg < 0:
            continue
        m = g_len[k] if g_len[k] < w_len[w] else w_len[w]
        gr = 0
        for j in range(m):
            if d[g_pos[k, j]] == w_pos[w, j]:
                gr += 1
        tot = 0
        for j in range(g_nd[k]):
            c = w_cnt[w, d[g_dl[k, j]]]
            gc = g_dc[k, j]
            tot += c if c < gc else gc
        acc += abs(gr - fg) + abs(tot - fg - fb[k, i, 1])
    return acc


@njit(cache=True)
def _tuple_violation(d, t, g_pos, g_len, g_dl, g_dc, g_nd, K, fb, w_pos, w_len, w_cnt):
    s = 0
    for i in range(t.shape[0]):
        s += _viol_direct(d, t[i], i, g_pos, g_len, g_dl, g_dc, g_nd, K, fb, w_pos, w_len, w_cnt)
    return s


@njit(cache=True)
def fit_decoder(d, t, active, steps, beta, patience, g_pos, g_len, g_dl, g_dc, g_nd, K, fb, w_pos, w_len, w_cnt):
    """Decoder given fixed target words: Metropolis transposition moves on the
    total violation of the N words (cheap: only N words are scored). Stops at
    zero, after `steps`, or after `patience` steps without a new best. d is left
    at the best decoder found; returns its total violation."""
    cur = _tuple_violation(d, t, g_pos, g_len, g_dl, g_dc, g_nd, K, fb, w_pos, w_len, w_cnt)
    if active.shape[0] == 0 or cur == 0:
        return cur
    best = cur
    best_d = d.copy()
    since = 0
    for _ in range(steps):
        a = active[np.random.randint(active.shape[0])]
        b = np.random.randint(25)
        if b >= a:
            b += 1
        d[a], d[b] = d[b], d[a]
        new = _tuple_violation(d, t, g_pos, g_len, g_dl, g_dc, g_nd, K, fb, w_pos, w_len, w_cnt)
        if new <= cur or np.random.random() < np.exp(-beta * (new - cur)):
            cur = new
        else:
            d[a], d[b] = d[b], d[a]
        if cur < best:
            best = cur
            best_d[:] = d
            since = 0
            if best == 0:
                return 0
        else:
            since += 1
            if since >= patience:
                break
    d[:] = best_d
    return best


@njit(cache=True)
def tuple_moves(d, t, active, moves, fit_steps, beta_fit, patience, beta_word, beta_accept, cap, stop_at_zero,
                g_pos, g_len, g_dl, g_dc, g_nd, K, fb, w_pos, w_len, w_cnt, lo, hi):
    """Word-anchored stochastic search over (decoder d, target words t).

    Each move picks a target i, draws a replacement word with probability
    proportional to exp(-beta_word * violation under the current d) (words with
    violation >= cap get weight 0), refits the decoder to the new words with
    `fit_decoder` (warm-started), and accepts if the total violation does not
    rise, or else with probability exp(-beta_accept * rise).
    d and t are updated in place. Returns the final total violation.

    Plain transposition MCMC on the marginal energy gets stuck once there are
    ~7 guesses; given the right words, though, fitting the decoder is easy.
    So the search moves through word space and fits the decoder inside.
    """
    N = t.shape[0]
    V = fit_decoder(d, t, active, fit_steps, beta_fit, patience, g_pos, g_len, g_dl, g_dc, g_nd, K, fb, w_pos, w_len, w_cnt)
    d2 = d.copy()
    t2 = t.copy()
    for _ in range(moves):
        if stop_at_zero and V == 0:
            break
        i = np.random.randint(N)
        n = hi[i] - lo[i]
        if n <= 1:
            continue
        dg, ddl = _decode_guesses(d, g_pos, g_len, g_dl, g_nd, K)
        wts = np.empty(n)
        tot = 0.0
        for r in range(n):
            v = _word_violation(lo[i] + r, dg, ddl, g_len, g_dc, g_nd, K, fb, i, w_pos, w_len, w_cnt, cap)
            x = np.exp(-beta_word * v) if v < cap else 0.0
            wts[r] = x
            tot += x
        if tot == 0.0:
            r = np.random.randint(n)
        else:
            u = np.random.random() * tot
            acc = 0.0
            r = 0
            while r < n - 1:
                acc += wts[r]
                if acc > u:
                    break
                r += 1
        d2[:] = d
        t2[:] = t
        t2[i] = lo[i] + r
        V2 = fit_decoder(d2, t2, active, fit_steps, beta_fit, patience, g_pos, g_len, g_dl, g_dc, g_nd, K, fb, w_pos, w_len, w_cnt)
        if V2 <= V or np.random.random() < np.exp(-beta_accept * (V2 - V)):
            d[:] = d2
            t[:] = t2
            V = V2
    return V


# -- guess selection ------------------------------------------------------------

@njit(cache=True)
def _score_counts(g_pos, g_len, g_dl, g_dc, g_nd, h_pos, h_len, h_cnt):
    m = g_len if g_len < h_len else h_len
    gr = 0
    for j in range(m):
        if g_pos[j] == h_pos[j]:
            gr += 1
    tot = 0
    for j in range(g_nd):
        c = h_cnt[g_dl[j]]
        gc = g_dc[j]
        tot += c if c < gc else gc
    return gr, tot


@njit(cache=True)
def feedback_entropy(c_pos, c_len, c_dl, c_dc, c_nd, h_pos, h_len, h_cnt, h_w):
    """Entropy (bits) of the joint feedback each candidate guess would produce.

    c_*: G candidate guesses (cipher-space letters, i.e. as typed).
    h_*: S hypotheses x U unsolved targets of *ciphered* target strings.
    h_w: hypothesis weights (sum to 1).
    Feedback is deterministic given a hypothesis, so this is the expected
    information gain of each guess under the hypothesis sample.
    """
    G = c_pos.shape[0]
    S, U = h_len.shape[0], h_len.shape[1]
    H = np.zeros(G)
    codes = np.empty(S, dtype=np.int64)
    for g in range(G):
        for s in range(S):
            code = 0
            for u in range(U):
                gr, tot = _score_counts(c_pos[g], c_len[g], c_dl[g], c_dc[g], c_nd[g],
                                        h_pos[s, u], h_len[s, u], h_cnt[s, u])
                code = code * FB_BASE + gr * 11 + tot
            codes[s] = code
        order = np.argsort(codes)
        h = 0.0
        run = 0.0
        for r in range(S):
            run += h_w[order[r]]
            if r == S - 1 or codes[order[r + 1]] != codes[order[r]]:
                if run > 0:
                    h -= run * np.log2(run)
                run = 0.0
        H[g] = h
    return H
