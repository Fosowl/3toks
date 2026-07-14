"""Accepted candidate for is_palindrome
Source URL: https://codereview.stackexchange.com/questions/236360/python-palindrome-program
License detection: NOT PERFORMED (future work — see REPORT.md safety section)
"""
def is_palindrome(x):
    '''
    True if the letters of x are the same when
    reversed, ignoring differences of case.
    >>> is_palindrome('')
    True
    >>> is_palindrome('a')
    True
    >>> is_palindrome('a0')
    False
    >>> is_palindrome('ab')
    False
    >>> is_palindrome('ab a')
    False
    >>> is_palindrome('Aba')
    True
    >>> is_palindrome('Àbà')
    True
    '''
    x = x.lower()
    # Alternative (letters only):
    #     x = ''.join(filter(str.isalpha, x.lower()))
    # Alternative (letters and digits only):
    #     x = ''.join(filter(str.isalnum x.lower()))
    return x[::-1] == x