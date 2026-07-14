"""Accepted candidate for levenshtein
Source URL: https://www.askpython.com/python/examples/levenshtein-python-troubleshooting-install
License detection: NOT PERFORMED (future work — see REPORT.md safety section)
"""
def levenshtein(str1, str2):
    """Calculating the Levenshtein distance between two strings."""
    n_m = [[0 for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
    for i in range(len(str1) + 1):
        n_m[i][0] = i
    for j in range(len(str2) + 1):
        n_m[0][j] = j
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                cost = 0
            else:
                cost = 1
            n_m[i][j] = min(n_m[i - 1][j] + 1,
                            n_m[i][j - 1] + 1,
                            n_m[i - 1][j - 1] + cost)
    return n_m[-1][-1]