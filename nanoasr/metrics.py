def _edit_distance(ref: list, hyp: list) -> int:
    """Standard Levenshtein edit distance via dynamic programming."""
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]


def word_error_rate(references: list[str], hypotheses: list[str]) -> float:
    """Compute WER over a corpus. Returns 0.0-1.0+ (can exceed 1 with many insertions)."""
    total_words = 0
    total_errors = 0
    for ref, hyp in zip(references, hypotheses):
        ref_words = ref.split()
        hyp_words = hyp.split()
        total_errors += _edit_distance(ref_words, hyp_words)
        total_words += len(ref_words)
    return total_errors / max(total_words, 1)


def char_error_rate(references: list[str], hypotheses: list[str]) -> float:
    """Compute CER over a corpus."""
    total_chars = 0
    total_errors = 0
    for ref, hyp in zip(references, hypotheses):
        total_errors += _edit_distance(list(ref), list(hyp))
        total_chars += len(ref)
    return total_errors / max(total_chars, 1)
