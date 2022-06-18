from helper_functions import *
import matplotlib.pyplot as plt
from stage2 import *

paths = []
for root, dirs, files in os.walk('displacement_data', ):
    for file in files:
        if file.endswith(".mat"):
            paths.append(os.path.join(root, file))

for pth in paths:
    displacement, tfm_ecg2, t_radar, fs_radar, t_ecg, fs_ecg = data_acquisition(pth)
    heart_rates_final, heart_rates_ref = compute_heart_rates(displacement, tfm_ecg2, t_radar, fs_radar, t_ecg, fs_ecg)
    with open(os.path.join('heart_rates', pth.rstrip('.mat').split('\\')[-1] + '.pkl'), 'wb') as outp:
        pickle.dump({'heart_rates_final': heart_rates_final, 'heart_rates_ref': heart_rates_ref}, outp, pickle.HIGHEST_PROTOCOL)


