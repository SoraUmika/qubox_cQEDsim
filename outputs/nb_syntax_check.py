import ast
import json

p = r"E:\qubox\notebooks\post_cavity_experiment_context_SIM.ipynb"

with open(p, "r", encoding="utf-8") as f:
    nb = json.load(f)

errors = []
code_cells = 0
for i, cell in enumerate(nb.get("cells", []), 1):
    if cell.get("cell_type") != "code":
        continue
    code_cells += 1
    src = "".join(cell.get("source", []))
    try:
        ast.parse(src)
    except Exception as exc:
        errors.append((i, str(exc)))

print("code_cells", code_cells)
print("syntax_errors", len(errors))
for idx, msg in errors[:30]:
    print(idx, msg)
