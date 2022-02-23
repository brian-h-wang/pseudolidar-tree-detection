"""
Preprocess the labels for PointPillars training:

Set all bounding boxes to have the same z-coordinate and height.
Set height equal to the point cloud height, set z-coordinate equal to the mean of
point cloud z min and z max.
"""
from pathlib import Path
from pointpillars.dataset import ForestDataset

# Load config to get z coordinate min and max
# TODO for now, just hardcode in the z coordinate min and max
z_max = 10.0
z_min = -5.0

# Set all bounding boxes to have this height and z-coordinate
height = z_max - z_min
z_coord = (z_max + z_min)/2.


train_data_dir = Path("/home/bhw45/RTJ2_dataset/training")
labels_in_dir = train_data_dir / "label_2_original"
labels_out_dir = train_data_dir / "label_2"

if not labels_out_dir.exists():
    labels_out_dir.mkdir(parents=True)

files = labels_in_dir.glob("*.txt")
print("Writing files...")
for f in files:
    name = f.name  # gives e.g. '000008.txt'
    label_in_path = str(labels_in_dir / name)
    label_out_path = str(labels_out_dir / name)
    calib_path = label_out_path.replace('label_2', 'calib')
    calib = ForestDataset.read_calib(calib_path)
    bounding_boxes = ForestDataset.read_label(label_in_path, calib)
    out_lines = []
    first_line = True
    for bbox in bounding_boxes:
        # Set the height and z-coordinate of the bounding box
        bbox.center[2] = z_coord
        bbox.size[1] = height

        kitti_line = bbox.to_kitti_format(score=1.0)
        line_no_score = kitti_line[:-5]  # remove the score, since this is a training label
        # add line breaks
        if first_line:
            first_line = False
        else:
            line_no_score = '\n' + line_no_score
        out_lines.append(line_no_score)
    if Path(label_out_path).exists():
        raise FileExistsError("File already exists, exiting")
    with open(label_out_path, 'w') as out_file:
        out_file.writelines(out_lines)

print("Done!")
