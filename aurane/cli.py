"""
Command-line interface for Aurane.

Main entry point that delegates to the modular CLI.
"""

import sys

def main():
    """Main entry point for the CLI."""
    try:
        from .cli import main as modular_main
        return modular_main()
    except ImportError as e:
        print(f"Error loading modular CLI: {e}", file=sys.stderr)
        # Fallback to a very basic compile if everything fails
        print("Attempting basic fallback...")
        return basic_fallback()

def basic_fallback():
    """Very basic fallback if modular CLI fails to load."""
    import argparse
    from .compiler import compile_file
    
    parser = argparse.ArgumentParser(prog="aurane")
    subparsers = parser.add_subparsers(dest="command")
    
    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument("input")
    compile_parser.add_argument("output")
    compile_parser.add_argument("--backend", default="torch")
    
    args = parser.parse_args()
    if args.command == "compile":
        try:
            compile_file(args.input, args.output, backend=args.backend)
            print(f"Compiled {args.input} to {args.output}")
            return 0
        except Exception as e:
            print(f"Compilation failed: {e}")
            return 1
    return 1

if __name__ == "__main__":
    sys.exit(main())
