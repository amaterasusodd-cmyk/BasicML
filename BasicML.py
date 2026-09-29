import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

np.random.seed(42) # makes sure data is somewhat deterministic

# creates 100 "data samples"
# each sample has 3 features, red, green, and blue
n_samples = 100
red = np.random.randint(1,256,n_samples)
green = np.random.randint(1,256,n_samples)
blue = np.random.randint(1,256,n_samples)

X = np.column_stack([red,green,blue])
y = red + 2*green + 3*blue

# splits training data randomly
# 80% will be used to train, 20% to test
X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

# uses LinearRegression to fit a model that minimizes
# squared error
model = LinearRegression(n_jobs=-1)
model.fit(X_train,y_train)

# shows results of coefficients and intercept
# coefficients should be near [1,2,3]
# intercept should be near 0
print("COEFFICIENTS: ", model.coef_)
print("INTERCEPT: ",model.intercept_)

# shows predictions of model
# as well as error
y_pred = model.predict(X_test)
print("MEAN SQUARED ERROR: ",mean_squared_error(y_test,y_pred))
print("R^2 SCORE: ",r2_score(y_test,y_pred))

X_test_feature = X_test[:,0]
sorted_idx = np.argsort(X_test_feature)

# Creates a graph to show the actual data versus the
# predicted data in the test
plt.figure(figsize=(8,5))
plt.scatter(X_test_feature,y_test,label='ACTUAL Y',s=40)
plt.scatter(X_test_feature,y_pred,label='PREDICTED Y',marker='X',s=40)
plt.plot(
    X_test_feature[sorted_idx],
    y_pred[sorted_idx],
    label='REGRESSION LINE',
    linewidth=2
)

plt.xlabel("FEATURE RED")
plt.ylabel("TARGET Y")
plt.title("LINEAR REGRESSION MODEL VISUALIZATION (USING FEATURE RED)")
plt.legend()
plt.grid(True)
plt.show()