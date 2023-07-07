import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from skbonus.linear_model import LADRegression

# Read csv file
conc = np.genfromtxt("Conc.txt")
abso = np.genfromtxt("Abso.txt")

# Initialize r2 array, reshape to (m,1)
r2_values = np.zeros((abso.shape[0],1))

# Intializa slope array
slopes = np.zeros((abso.shape[0], 1))

# For each line of abso
for i in range(abso.shape[0]):
    
    df = pd.DataFrame({'Concentration': conc, 'Absorbance': abso[i,:]})
    df = df.dropna()
    
    if df.empty or len(df) < 2:
        r2_values[i] = 0
    else:
        # from df to np
        x = df['Concentration'].values.reshape(-1,1)
        y = df['Absorbance'].values
        
        # LAD linear regressuib, pass origin point
        model = LADRegression(fit_intercept=False)
        model.fit(x, y)
    
        # R2
        r2_values[i] = r2_score(y, model.predict(x))
        
        # 保存斜率
        slopes[i] = model.coef_
    
# save result
np.savetxt("r2_values.txt", r2_values, delimiter=",")
np.savetxt('Slope.txt', slopes)

print("done!")