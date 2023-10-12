import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from skbonus.linear_model import LADRegression
from joblib import Parallel, delayed
from tqdm import tqdm

# Read csv file
conc = np.genfromtxt("Conc.txt")
abso = np.genfromtxt("Abso.txt")

# Initialize r2 array, reshape to (m,1)
r2_values = np.zeros((abso.shape[0],1))

# Intializa slope array
slopes = np.zeros((abso.shape[0], 1))

# Define a function for the computation that needs to be parallelized
def compute(i):
    df = pd.DataFrame({'Concentration': conc, 'Absorbance': abso[i,:]})
    df = df.dropna()
    
    if df.empty or len(df) < 2:
        return 0, 0
    else:
        # from df to np
        x = df['Concentration'].values.reshape(-1,1)
        y = df['Absorbance'].values
        
        # LAD linear regression, pass origin point
        model = LADRegression(fit_intercept=False)
        model.fit(x, y)
    
        # R2
        r2 = r2_score(y, model.predict(x))
        
        # Save slope
        slope = model.coef_
        
        return r2, slope

# Use joblib to parallelize the loop
results = Parallel(n_jobs=-1)(delayed(compute)(i) for i in tqdm(range(abso.shape[0]))) # n_jobs=-1， use all threads

# Unpack results
r2_values, slopes = zip(*results)

# Save result
np.savetxt("r2_values_multi.txt", r2_values, delimiter=",")
np.savetxt('Slope_multi.txt', slopes)

print("done!")
