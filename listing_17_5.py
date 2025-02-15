import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt

signal, bkps = rpt.pw_constant(n_samples=1000, n_features=1, n_bkps=4, 
                               noise_std=4, seed=42) 

model = rpt.Pelt(model="rbf") 
model.fit(signal) 

result = model.predict(pen=10) 

rpt.display(signal, [], result) 
plt.show()
