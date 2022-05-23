from helper_functions import *
import matplotlib.pyplot as plt
from stage2 import *

paths = []
for root, dirs, files in os.walk('displacement_data', ):
    for file in files:
        if file.endswith(".mat"):
            paths.append(os.path.join(root, file))

for path in paths:
    displacement, tfm_ecg2, t_radar, fs_radar, t_ecg, fs_ecg = data_acquisition(path)


compute_heart_rates()
