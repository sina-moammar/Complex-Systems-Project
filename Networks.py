import numpy as np
from Tools import power_law_HP_model


data_dir = '/mnt/extra/University/10 - Complex Systems/Project/data'


power_law_HP_model(100000, 2.01, 4, -0.2, data_dir)
power_law_HP_model(100000, 2.2, 4, -0.08, data_dir)
power_law_HP_model(100000, 2.4, 4, -0.03, data_dir)
power_law_HP_model(100000, 2.6, 4, -0.03, data_dir)
power_law_HP_model(100000, 2.8, 4, -0.028, data_dir)
power_law_HP_model(100000, 3, 4, -0.05, data_dir)
power_law_HP_model(100000, 5, 4, -0.5, data_dir)
power_law_HP_model(100000, np.inf, 4, 0, data_dir)
