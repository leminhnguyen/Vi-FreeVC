
files = [
    "filelists/test-vi.txt",
    "filelists/train-vi.txt",
    "filelists/val-vi.txt",
]
for file in files:
    lines = open(file).read().split("\n")
    keeps = []
    for line in lines:
        if "hn_mp_vdts" in line:
            keeps.append(line)
    open(file.replace(".txt", "-hn_mp_vdts.txt"), "w").write("\n".join(keeps))