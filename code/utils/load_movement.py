# %%
import numpy as np
import os
import pandas as pd
from pathlib import Path

# %%
infolder = Path('/Users/joebathelt/Library/Containers/io.mountainduck/Data/Library/Application Support/Mountain Duck/Volumes.noindex/HCP.localized/hcp-openaccess/HCP_1200/')
subject_list = sorted([f for f in os.listdir(infolder) if os.path.isdir(infolder / f)])

# %%
movement_data = []
for i, subject in enumerate(subject_list):
    print(f'Processing subject {i+1} of {len(subject_list)}', end='\r')
    try:
        movement_folder = infolder / subject / 'MNINonLinear/Results'
        movement_subfolders = [f for f in os.listdir(movement_folder) if f.startswith('rfMRI_REST')]
        for subfolder in movement_subfolders:
            movement_files = [f for f in os.listdir(movement_folder / subfolder) if f.endswith('Movement_RelativeRMS_mean.txt')]
            subject_data = []
            for movement_file in movement_files:
                data = pd.read_csv(movement_folder / subfolder / movement_file, header=None)[0].values[0]
                subject_data.append(data)
    except:
        subject_data = []

    movement_data.append({
        'Subject': subject,
        'Movement_RelativeRMS_mean': np.mean(subject_data)
    })

pd.DataFrame(movement_data).to_csv('movement_data.csv', index=False)
# %%
