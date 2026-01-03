import nbformat

def extract_code(notebook_path):
    with open(notebook_path, 'r') as f:
        nb = nbformat.read(f, as_version=4)
    
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            print(f"\n--- CELL {i} ---\n")
            print(cell.source)

if __name__ == "__main__":
    extract_code("/Users/yuyamukai/dev/tunix/grpo-demo-gemma2-2b.ipynb")
