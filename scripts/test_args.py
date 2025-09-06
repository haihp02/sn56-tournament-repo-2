#!/usr/bin/env python3
import sys

def main():
    print("Script name:", sys.argv[0])
    print("Number of arguments:", len(sys.argv) - 1)
    print()
    
    if len(sys.argv) > 1:
        print("All arguments:")
        for i, arg in enumerate(sys.argv[1:], 1):
            print(f"  Argument {i}: {arg}")
    else:
        print("No arguments provided")

if __name__ == "__main__":
    main()
