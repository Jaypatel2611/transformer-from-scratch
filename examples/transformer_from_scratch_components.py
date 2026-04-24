"""
Educational demonstration of transformer components.

This script runs the educational walkthrough of transformer internals
by executing src/from_scratch/transformer_from_scratch.py

Run with: python examples/transformer_from_scratch_components.py
Or: python -m examples.transformer_from_scratch_components
"""

if __name__ == "__main__":
    # Import and run the educational module
    # We use runpy to execute the module safely without adding to sys.path
    import runpy
    import sys
    from pathlib import Path
    
    # Get the path to the educational script
    script_path = Path(__file__).parent.parent / "src" / "from_scratch" / "transformer_from_scratch.py"
    
    # Execute it as a script
    runpy.run_path(str(script_path), run_name="__main__")
    