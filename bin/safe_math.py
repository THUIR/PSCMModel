import sys, os, re, math, random

def safe_log(num, base):
    if num <= 1e-6:
        return -20.0
    if num >= 1.0:
        return 0.0
    return math.log(num, base)