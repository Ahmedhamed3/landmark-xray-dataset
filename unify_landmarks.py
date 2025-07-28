import json
import csv
import glob
import os

folders = [str(i) for i in range(1, 20)]
rows = []
for folder in folders:
    patterns = [
        os.path.join(folder, 'IAC*mrk.json'),
        os.path.join(folder, 'LBM*mrk.json'),
        os.path.join(folder, 'MF*mrk.json'),
        os.path.join(folder, 'Mid-symphyseal*mrk.json'),
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    for path in files:
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue
        cps = data.get('markups', [{}])[0].get('controlPoints', [])
        if not cps:
            continue
        for cp in cps:
            pos = cp.get('position', [])
            if len(pos) >= 2:
                label = cp.get('label', os.path.basename(path))
                x, y = pos[0], pos[1]
                rows.append([folder, label, x, y])

with open('important_landmarks.csv', 'w', newline='') as out:
    writer = csv.writer(out)
    writer.writerow(['SampleID', 'LandmarkLabel', 'X', 'Y'])
    writer.writerows(rows)
