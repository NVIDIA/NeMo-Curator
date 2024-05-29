import sys

def is_float(s):
    try:
        float(s)
    except ValueError:
        return False
    return True

print(len([w for w in sys.stdin.read().split() if not is_float(w)]))
