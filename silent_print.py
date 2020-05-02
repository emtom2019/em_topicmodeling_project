import os, sys

class SilentPrinting:
    def __init__(self, verbose=False):
        self.verbose=verbose
        
    def __enter__(self):
        if not self.verbose:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.verbose:
            sys.stdout.close()
            sys.stdout = self._original_stdout

if __name__ == "__main__":
    print("Testing Silent printer - This should print")
    with SilentPrinting():
        print("this should NOT print")
    print("This should print")