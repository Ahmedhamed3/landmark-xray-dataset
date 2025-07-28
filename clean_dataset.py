import csv
import json
import glob
import os


def load_sample(folder):
    """Load landmarks for one sample folder."""
    landmarks = []
    image_path = None
    for ext in ('*.jpg', '*.png'):
        imgs = glob.glob(os.path.join(folder, ext))
        if imgs:
            image_path = imgs[0]
            break
    for path in glob.glob(os.path.join(folder, '*.mrk.json')):
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue
        markups = data.get('markups', [])
        if not markups:
            continue
        m = markups[0]
        cps = m.get('controlPoints', [])
        measurements = {m.get('name', '').lower(): m.get('value') for m in m.get('measurements', []) if 'value' in m}
        for cp in cps:
            pos = cp.get('position', [])
            if len(pos) < 2:
                continue
            label = cp.get('label', '').strip()
            rec = {
                'PointLabel': label,
                'X': pos[0],
                'Y': pos[1]
            }
            if 'angle' in measurements:
                rec['Angle'] = measurements['angle']
            if 'length' in measurements:
                rec['Distance'] = measurements['length']
            landmarks.append(rec)
    return image_path, landmarks


def main():
    folders = [str(i) for i in range(1, 20)]
    sample_data = {}
    label_sets = []
    for folder in folders:
        img, marks = load_sample(folder)
        sample_data[folder] = {'image': img, 'landmarks': marks}
        label_sets.append({m['PointLabel'] for m in marks})

    if not label_sets:
        print('No landmarks found')
        return

    common_labels = set.intersection(*label_sets)
    rows = []
    for folder in folders:
        info = sample_data[folder]
        for mark in info['landmarks']:
            if mark['PointLabel'] not in common_labels:
                continue
            row = {
                'SampleID': folder,
                'ImagePath': info['image'],
                'PointLabel': mark['PointLabel'],
                'X': mark['X'],
                'Y': mark['Y'],
                'Angle': mark.get('Angle', ''),
                'Distance': mark.get('Distance', '')
            }
            rows.append(row)

    out_path = 'clean_landmarks.csv'
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['SampleID', 'ImagePath', 'PointLabel', 'X', 'Y', 'Angle', 'Distance'])
        writer.writeheader()
        writer.writerows(rows)

    print(f'Wrote {len(rows)} rows to {out_path}')


if __name__ == '__main__':
    main()
