try:
    from physiolabxr import physiolabxr
except ImportError:
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from physiolabxr import physiolabxr

if __name__ == '__main__':
    physiolabxr()