import json
from pathlib import Path

NB = Path(r"c:\Users\jl82323\Box\Shyam Shankar Quantum Circuits Group\Users\Users_JianJun\cQED_simulation\post_cavity_experiment_context_SIM.ipynb")

with NB.open("r", encoding="utf-8") as f:
    data = json.load(f)

changed = 0
for cell in data.get("cells", []):
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", []))
    marker = 'if MODE == "hardware":\n'
    if marker not in src:
        continue
    # Keep only simulation-side behavior, drop embedded hardware code text completely.
    cell["source"] = [
        "print('[SIM] Hardware path removed in this local simulation-only notebook.')\n"
    ]
    changed += 1

with NB.open("w", encoding="utf-8") as f:
    json.dump(data, f, indent=1)

print(f"updated_cells={changed}")
print(str(NB))
