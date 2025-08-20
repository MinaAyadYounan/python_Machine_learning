import numpy as np
def sigmoid(z):
    return 1.0 /(1.0 + np.exp(-z))
def calc_gradient(theta, x, y): 
     m = y.size
     return (x.T @ (sigmoid(x @ theta) - y)) / m
def gradient_descent(x,y,alpha = 0.1 ,num_iteration = 100,tolerance = 1e-7) :
     x_b = np.c_[np.ones((x.shape[0],1)), x]
     theta = np.zeros(x_b.shape[1])
     for i in range(num_iteration):
          grad = calc_gradient(theta,x_b,y)
          theta -= alpha*grad 
          if np.linalg.norm(grad) < tolerance :
               break
     return theta
     
def predict_prob (x,theta):
     x_b = np.c_[np.ones((x.shape[0],1)), x]
     return sigmoid(x_b @ theta)
def predict(x,theta,thershold = 0.5):
     return (predict_prob(x,theta) >= thershold).astype(int)
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
x,y = load_breast_cancer(return_X_y=True)
# the data we load for cancer breast each row represent a paitent 
# and each column represent = a measured feature of that tumor (like 
# “mean radius”, “mean texture”, “mean smoothness”… about 30 features in total).
# our model predict the probability that this tumor is malignant
x_train,x_test ,y_train ,y_test = train_test_split(x,y,test_size=0.2)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
theta_hat = gradient_descent(x_train_scaled,y_train,alpha=0.1)
y_pred_train = predict(x_train_scaled,theta_hat)
y_pred_test = predict(x_test_scaled,theta_hat)
train_acc = accuracy_score(y_train,y_pred_train)
test_acc =accuracy_score(y_test,y_pred_test)
print(train_acc)
print(test_acc)
print(len(theta_hat))
print(len(y_pred_train))