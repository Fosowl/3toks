"""Accepted candidate for caesar_encode
Source URL: https://caesarcipher.org/learn/python-caesar-cipher-complete-programming-tutorial-with-source-code
License detection: NOT PERFORMED (future work — see REPORT.md safety section)
"""
def caesar_encode(text, shift):
    """
    Encrypt text using Caesar cipher with specified shift.
    
    Args:
        text (str): The plaintext to encrypt
        shift (int): Number of positions to shift (positive for right, negative for left)
    
    Returns:
        str: The encrypted ciphertext
    
    Example:
        >>> caesar_encrypt("HELLO WORLD", 3)
        'KHOOR ZRUOG'
    """
    # Input validation
    if not isinstance(text, str):
        raise TypeError("Text must be a string")
    if not isinstance(shift, int):
        raise TypeError("Shift must be an integer")
    
    # Normalize shift to valid range (0-25)
    shift = shift % 26
    
    result = []
    
    for char in text:
        if char.isalpha():
            # Determine if uppercase or lowercase
            is_upper = char.isupper()
            
            # Convert to uppercase for calculation
            char_upper = char.upper()
            
            # Calculate shifted position
            char_position = ord(char_upper) - ord('A')
            shifted_position = (char_position + shift) % 26
            
            # Convert back to character
            new_char = chr(shifted_position + ord('A'))
            
            # Maintain original case
            result.append(new_char if is_upper else new_char.lower())
        else:
            # Preserve non-alphabetic characters (spaces, punctuation, numbers)
            result.append(char)
    
    return ''.join(result)