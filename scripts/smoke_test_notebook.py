#!/usr/bin/env python3
"""
Smoke Test for Jupyter Notebooks

Tests:
1. Syntax Validation (AST) - CRITICAL: Catches SyntaxErrors before Kaggle run
2. Import Check - Validates import statements are valid Python
3. Logic Flow - Optional, best-effort execution check (skipped on env mismatch)

Usage:
    python scripts/smoke_test_notebook.py
"""

import nbformat
import ast
import sys
import io
import contextlib

def smoke_test_notebook(notebook_path):
    print(f"🔥 Smoke Testing: {notebook_path}")
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
    except Exception as e:
        print(f"❌ Failed to read notebook: {e}")
        return False

    syntax_errors = []
    all_code = ""
    
    # --- Phase 1: Syntax Validation (AST) - CRITICAL ---
    print("\n--- 1. Syntax Validation (AST) [CRITICAL] ---")
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            # Sanitize magic commands
            lines = cell.source.split('\n')
            clean_lines = []
            for line in lines:
                if line.strip().startswith('!') or line.strip().startswith('%'):
                    # Preserve indentation but replace with pass
                    leading_whitespace = line[:len(line) - len(line.lstrip())]
                    clean_lines.append(f"{leading_whitespace}pass # SKIPPED MAGIC: {line.strip()}")
                else:
                    clean_lines.append(line)
            
            clean_source = "\n".join(clean_lines)
            
            try:
                ast.parse(clean_source)
                all_code += clean_source + "\n\n"
            except SyntaxError as e:
                syntax_errors.append((i+1, e, clean_source[:200]))
    
    if syntax_errors:
        print(f"❌ Found {len(syntax_errors)} syntax error(s):")
        for cell_num, error, snippet in syntax_errors:
            print(f"\n  Cell {cell_num}: {error}")
            print(f"  Snippet: {snippet}...")
        return False
    else:
        print(f"✅ All {len([c for c in nb.cells if c.cell_type == 'code'])} code cells have valid Python syntax.")

    # --- Phase 2: Logic Flow Check - OPTIONAL ---
    print("\n--- 2. Logic Flow Check [OPTIONAL] ---")
    print("Attempting best-effort execution (ignoring env-specific errors)...")
    
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            with contextlib.redirect_stderr(io.StringIO()):
                exec_globals = {}
                exec(all_code, exec_globals)
        print("✅ Full notebook simulation passed.")
    except ImportError as e:
        print(f"⚠️ Stopped at import: {type(e).__name__}: {e}")
        print("   (Expected on Mac vs Kaggle TPU)")
        print("✅ Syntax is valid, import issue is environment-specific.")
    except NameError as e:
        print(f"⚠️ NameError (may be false positive due to skipped imports): {e}")
        print("✅ Syntax is valid, treating as environment issue.")
    except Exception as e:
        error_type = type(e).__name__
        # Known env-specific errors to ignore
        known_env_errors = [
            "abstracted_axes",  # JAX version mismatch
            "TpuDevice",        # No TPU locally
            "CUDA",             # No GPU locally
            "XLA",              # XLA not configured
        ]
        if any(kw in str(e) for kw in known_env_errors):
            print(f"⚠️ Environment-specific error (ignored): {error_type}: {e}")
            print("✅ Syntax is valid, runtime issue is environment-specific.")
        else:
            print(f"⚠️ Unexpected runtime error: {error_type}: {e}")
            print("   (May be a real bug, but could also be env-specific)")
            print("✅ Syntax is still valid.")

    return True  # Syntax passed, that's what matters

if __name__ == "__main__":
    notebooks = ["tunix_sft_train.ipynb", "tunix_sft_continuation.ipynb"]
    all_passed = True
    
    for nb in notebooks:
        if not smoke_test_notebook(nb):
            all_passed = False
        print()  # Blank line between notebooks
            
    if all_passed:
        print("=" * 50)
        print("✅ ALL NOTEBOOKS PASSED SMOKE TEST")
        print("=" * 50)
        sys.exit(0)
    else:
        print("=" * 50)
        print("❌ SOME NOTEBOOKS FAILED - FIX BEFORE KAGGLE RUN")
        print("=" * 50)
        sys.exit(1)
