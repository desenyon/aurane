"""
Command-line interface for Aurane.

Provides commands to compile and run Aurane programs.
This module now delegates to cli_enhanced for better UX when rich is available.
"""

import sys

# Try to use enhanced CLI if available
try:
    from .cli_enhanced import main as enhanced_main

    USE_ENHANCED = True
except ImportError:
    USE_ENHANCED = False

if USE_ENHANCED:
    # Use the enhanced CLI with rich formatting
    def main():
        """Main entry point - delegates to enhanced CLI."""
        return enhanced_main()

else:
    # Fallback to basic CLI
    import argparse
    import subprocess
    from pathlib import Path
    from .compiler import compile_file, compile_to_temp, CompilationError

    def cmd_compile(args):
        """Handle the 'compile' command."""
        try:
            compile_file(args.input, args.output, backend=args.backend)
            return 0
        except CompilationError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Unexpected error: {e}", file=sys.stderr)
            return 1

    def cmd_run(args):
        """Handle the 'run' command."""
        try:
            # Read the source file
            input_file = Path(args.input)
            if not input_file.exists():
                print(f"Error: Source file not found: {args.input}", file=sys.stderr)
                return 1

            source = input_file.read_text(encoding="utf-8")

            # Compile to temporary file
            temp_path = compile_to_temp(source, backend=args.backend)

            print(f"Compiled to temporary file: {temp_path}")
            print("Running...")
            print("-" * 60)

            # Run the generated Python file
            result = subprocess.run([sys.executable, str(temp_path)], cwd=input_file.parent)

            # Clean up temporary file
            if not args.keep_temp:
                temp_path.unlink()
            else:
                print("-" * 60)
                print(f"Temporary file kept at: {temp_path}")

            return result.returncode

        except CompilationError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Unexpected error: {e}", file=sys.stderr)
            return 1

    def main():
        """Main entry point for the CLI."""
        parser = argparse.ArgumentParser(
            prog="aurane", description="Aurane ML DSL - Transpile .aur files to Python"
        )

        subparsers = parser.add_subparsers(dest="command", help="Command to execute")

        # Compile command
        compile_parser = subparsers.add_parser("compile", help="Compile an Aurane file to Python")
        compile_parser.add_argument("input", help="Input .aur file")
        compile_parser.add_argument("output", help="Output .py file")
        compile_parser.add_argument(
            "--backend",
            default="torch",
            choices=["torch"],
            help="Code generation backend (default: torch)",
        )

        # Run command
        run_parser = subparsers.add_parser("run", help="Compile and run an Aurane file")
        run_parser.add_argument("input", help="Input .aur file")
        run_parser.add_argument(
            "--backend",
            default="torch",
            choices=["torch"],
            help="Code generation backend (default: torch)",
        )
        run_parser.add_argument(
            "--keep-temp", action="store_true", help="Keep the temporary compiled Python file"
        )

        # Parse arguments
        args = parser.parse_args()

        # Execute command
        if args.command == "compile":
            return cmd_compile(args)
        elif args.command == "run":
            return cmd_run(args)
        else:
            parser.print_help()
            return 1


if __name__ == "__main__":
    sys.exit(main())
