"""Accepted candidate for is_prime
Source URL: https://stackoverflow.com/questions/15285534/isprime-function-for-python-language
License detection: NOT PERFORMED (future work — see REPORT.md safety section)
"""
def is_prime(n):
    for i in range(2,int(n**0.5)+1):
        if n%i==0:
            return False
        
    return True