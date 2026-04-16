import json

with open("/home/saij/ml/numin/train.ipynb", "r") as f:
    nb = json.load(f)

print("".join(nb["cells"][12]["source"]))
