import json

notebook_path = 'code/_eda_4_machine_learning.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        print(f"\n# Cell {i}")
        source = "".join(cell['source'])
        print(source)
