lines = []

for i in range(1, 35):
    lines.append(f"{i:04d} 1")

for i in range(35, 69):
    lines.append(f"{i:04d} 0")

for i in range(69, 130):
    lines.append(f"{i:04d} 1")

for i in range(130, 189):
    lines.append(f"{i:04d} 0")

with open("data/label/labels.txt", "w") as f:
    for line in lines:
        f.write(line + "\n")

