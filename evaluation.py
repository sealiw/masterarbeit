from helper_functions import *
import matplotlib.pyplot as plt
from stage2 import *

displacement, tfm_ecg2, t_radar, fs_radar, t_ecg, fs_ecg = data_acquisition()


compute_heart_rates()
