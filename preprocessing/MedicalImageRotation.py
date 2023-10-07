import SimpleITK as sitk
import random
import math
import numpy as np

# Load the 3D medical image
image = sitk.ReadImage(r"D:\MyProject\Python\Extrajob\python_pakage\dist\CengLiang20200616Base.nii")

# Generate random rotation angles for X, Y, and Z axes in degrees
random_angle_x_degrees = 10
random_angle_y_degrees = 10
random_angle_z_degrees = 10

# Convert the angles to radians (SimpleITK uses radians)
random_angle_x_radians = math.radians(random_angle_x_degrees)
random_angle_y_radians = math.radians(random_angle_y_degrees)
random_angle_z_radians = math.radians(random_angle_z_degrees)

# Create a transformation
rotation_transform = sitk.Euler3DTransform()
rotation_transform.SetRotation(random_angle_x_radians, random_angle_y_radians, random_angle_z_radians)
rotation_transform.SetCenter(image.TransformContinuousIndexToPhysicalPoint([size/2 for size in image.GetSize()]))

# Apply the transformation to the image
rotated_image = sitk.Resample(image, rotation_transform)
# Save the deformed 3D image
sitk.WriteImage(rotated_image, "path_to_save_deformed_image.nii")