# %%
import nibabel as nib
import numpy as np
from pathlib import Path
import pandas as pd

# Set project paths
project_folder = Path('/Users/joebathelt/Documents/1_Projects/BrainCompensation/')
atlas_folder = project_folder / 'data/HCP_PTN1200/groupICA/'

for n_nodes in [15, 25, 50, 100, 200, 300]:
    atlas_file = atlas_folder / f'groupICA_3T_HCP1200_MSMAll_d{n_nodes}.ica/melodic_IC_sum.nii.gz'
    atlas = nib.load(atlas_file)

    # Extract data and affine transformation
    atlas_data = atlas.get_fdata()
    affine = atlas.affine  # This is the transformation matrix to MNI space

    results = []
    for i in range(atlas_data.shape[3]):
        component_image = atlas_data[:, :, :, i]

        # Find voxel coordinates of the maximum intensity
        max_voxel_coords = np.unravel_index(np.argmax(component_image, axis=None), component_image.shape)

        # Convert voxel coordinates to MNI space
        max_mni_coords = nib.affines.apply_affine(affine, max_voxel_coords)

        results.append({
            'component': f'IC{i}',
            'x': max_mni_coords[0],
            'y': max_mni_coords[1],
            'z': max_mni_coords[2],
        })

    # Save results
    pd.DataFrame(results).to_csv(atlas_folder / f'groupICA_3T_HCP1200_MSMAll_d{n_nodes}.ica/melodic_IC_node_coords.csv', index=False)

print("Centroid coordinates saved in MNI space.")
# %%
