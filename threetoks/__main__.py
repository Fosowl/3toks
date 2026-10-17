"""Package entry point: ``python3 -m threetoks`` runs the CLI."""
import sys

from threetoks.launch import main

if __name__ == "__main__":
    sys.exit(main())
