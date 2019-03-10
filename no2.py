import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('iris.csv',header=None)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)
X = df.iloc[0:100, [0,1,2,3]].values
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()

class Perceptron(object):
   def __init__(self, rate = 0.1, niter = 10):
      self.rate = rate
      self.niter = niter

   def fit(self, X, y):
      self.weight = np.zeros(1 + X.shape[1])
      self.weight[:5]=0.5 #bias=weight1-4=0.5

      # Number of misclassifications
      self.errors = []
      self.accuracy=[]
      
      for i in range(self.niter):
         err = 0
         sum1=sum2=0
         for xi, target in zip(X, y):
            err =(target-self.output(xi))*(target-self.output(xi))
            if self.predict(xi)==1 and target==1:
                sum1+=1 #jumlah True Positive
            elif self.predict(xi)==0 and target==0:
                sum2+=1 #jumlah True Negative
            accr=(sum1+sum2)/100 #accurancy
            delta_w = 2*xi*(self.output(xi)-target)*(1-self.output(xi))*self.output(xi) #delta weight
            delta_w0= 2*(self.output(xi)-target)*(1-self.output(xi))*self.output(xi) #delta bias
            self.weight[1:] = self.weight[1:]-self.rate*delta_w #weight baru
            self.weight[0] =self.weight[0]-self.rate*delta_w0 #bias baru
         self.errors.append(err)
         self.accuracy.append(accr) #disimpan ke dalam array accurancy
      return self

   def output(self, X):
       return np.dot(X, self.weight[1:]) + self.weight[0]

   def predict(self, X):
      """Return class label after unit step"""
      return np.where(self.output(X) >= 0.5, 1, 0)

#untuk grafik error
pn = Perceptron(0.1, 300)
pn.fit(X, y)
plt.plot(range(1, len(pn.errors) + 1), pn.errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()

#untuk grafik accuracy
pn = Perceptron(0.8, 300)
pn.fit(X, y)
plt.plot(range(1, len(pn.accuracy) + 1), pn.accuracy, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()