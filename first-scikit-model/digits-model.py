import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import linear_model

#Data Loading
digits = datasets.load_digits()
print(digits.images[0])

#Plotting
plt.figure()
plt.imshow(16 - digits.images[0])

#Split the labelled datasets to training set and test set
index_to_split = -1
x_train = digits.data[:index_to_split]
y_train = digits.target[:index_to_split]
x_test = digits.data[index_to_split:]
y_test = digits.target[index_to_split:]

#Train Model y = Xw
x_train_inv = np.linalg.pinv(x_train)
w = np.dot(x_train_inv, y_train)
print('w: {}'.format(w))

#Testing Model
y_hat = x_test @ w
print('y_hat: {}, rounds to: {}'.format(y_hat, int(np.round(y_hat)[0])))
print('Actual Y Label: {}'.format(y_test))
plt.figure()
plt.imshow(16 - digits.images[-1])

#Model with Scikit Learn
reg = linear_model.Ridge(alpha=0)
reg.fit(x_train, y_train)
y_hat = reg.predict(x_test)
print('y_hat: {}, rounds to: {}'.format(y_hat, int(np.round(y_hat)[0])))
print('Actual y label: {}'.format(y_test))
plt.figure()
plt.imshow(16 - digits.images[-1])
print('w: {}'.format(reg.coef_))
print('shape: {}'.format(reg.coef_.shape))