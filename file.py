import os

folder = 'data'

files = [
    'ch150.tsp.txt',
    'eil51.tsp.txt',
    'kroA100.tsp.txt',
    'kroB200.tsp.txt'
]

for file in files:

    old_path = os.path.join(folder, file)

    new_path = os.path.join(folder, file.replace('.txt', ''))

    os.rename(old_path, new_path)

    print(f'Renamed: {old_path} -> {new_path}')