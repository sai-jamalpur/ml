import json

with open("/home/saij/ml/numin/train.ipynb", "r") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        src = "".join(cell.get("source", []))
        if "PortfolioDataset" in src or "portfolio" in src:
            print(f"Cell {i}:\n{src}\n------------------")
