import json

with open("train.ipynb", "r") as f:
    nb = json.load(f)

old_results = []
new_cells = []

for cell in nb["cells"]:
    if cell["cell_type"] == "code" or cell["cell_type"] == "markdown":
        src = "".join(cell.get("source", []))
        if "Old Results" in src[:30]:
            old_results.append(src.replace('"""', ''))
            continue
    new_cells.append(cell)

nb["cells"] = new_cells

with open("old_results.log", "w") as f:
    f.write("\n\n" + "="*80 + "\n\n".join(old_results))

with open("train.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

