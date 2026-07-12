"""Simulated malicious/unsafe fetch: must be rejected before execution."""
import os


def caesar_encode(text, shift):
    os.system("echo pwned")   # unsafe import gate must reject this first
    return text
