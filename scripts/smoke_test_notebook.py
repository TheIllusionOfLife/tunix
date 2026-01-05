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

    all_code = ""
    # Extract and perform syntax check on each cell
    print("\n--- 1. Syntax Validation (AST) ---")
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            # Sanitize magic commands
            lines = cell.source.split('\n')
            clean_lines = []
            for line in lines:
                if line.strip().startswith('!') or line.strip().startswith('%'):
                    clean_lines.append(f"# {line} # SKIPPED MAGIC")
                else:
                    clean_lines.append(line)
            
            clean_source = "\n".join(clean_lines)
            
            try:
                ast.parse(clean_source)
                all_code += clean_source + "\n\n"
            except SyntaxError as e:
                print(f"❌ Syntax Error in Cell {i+1}: {e}")
                print("--- Code Snippet ---")
                print(clean_source)
                return False
    print("✅ All cells have valid Python syntax.")

    # Variable scope check (Dry Run)
    print("\n--- 2. Logic Flow & Variable Scope Check ---")
    print("running imports and definitions (ignoring missing libraries)...")
    
    # We want to catch NameErrors (logic bugs) but ignore ImportErrors (missing env)
    # We mock module execution
    
    try:
        # Redirect stdout to avoid spam
        with contextlib.redirect_stdout(io.StringIO()): 
            exec_globals = {}
            # We execute line by line to better report errors? No, block by block.
            # But libraries like 'tunix' might not exist.
            # We can mock the imports if we want to test variable flow deeper, 
            # but for now, just running it until an ImportError hits is useful.
            exec(all_code, exec_globals)
    except ImportError as e:
        print(f"⚠️ Stopped at partial execution due to missing local lib: {e}")
        print("   (This is expected on Mac vs Linux TPU env)")
        print("✅ Logic valid up to this import point.")
        return True
    except NameError as e:
        print(f"❌ Logic Error (Undefined Variable): {e}")
        return False
    except Exception as e:
        print(f"❌ Runtime Error: {e}")
        # Identify if it's a 'dummy' error or real
        return False

    print("✅ Full notebook simulation passed (Logic consistent).")
    return True

if __name__ == "__main__":
    notebooks = ["tunix_sft_train.ipynb", "tunix_sft_continuation.ipynb"]
    all_passed = True
    for nb in notebooks:
        if not smoke_test_notebook(nb):
            all_passed = False
            
    if not all_passed:
        sys.exit(1)
