"""
Preprocess the labels for PointPillars training:

Set all bounding boxes to have the same z-coordinate and height.
Set height equal to the point cloud height, set z-coordinate equal to the mean of
point cloud z min and z max.
"""

# Load config to get z coordinate min and max
# TODO for now, just hardcode in the z coordinate min and max

