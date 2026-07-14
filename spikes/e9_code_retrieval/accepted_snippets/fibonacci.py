"""Accepted candidate for fibonacci
Source URL: https://waynelambert.dev/blog/post/fibonacci-sequence-algorithm-python/
License detection: NOT PERFORMED (future work — see REPORT.md safety section)
"""
def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    elif n > 1:
        return fibonacci(n - 1) + fibonacci(n - 2)