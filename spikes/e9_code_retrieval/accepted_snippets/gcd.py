"""Accepted candidate for gcd
Source URL: https://stackoverflow.com/questions/11175131/code-for-greatest-common-divisor-in-python
License detection: NOT PERFORMED (future work — see REPORT.md safety section)
"""
def gcd(x, y):
    while y != 0:
        (x, y) = (y, x % y)
    return x