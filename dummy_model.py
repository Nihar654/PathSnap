import joblib, numpy as np
from sklearn.ensemble import RandomForestRegressor
import os

# generate dummy data with 5 features
X = np.random.rand(20, 5)
y = np.random.rand(20)

model = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y)

# Ensure the shared submission folder exists
os.makedirs('submission', exist_ok=True)

# Save to the shared submission folder (relative path)
joblib.dump(model, r'submission\\model.pkl')
print('dummy model saved')