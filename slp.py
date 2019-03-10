import pandas as pd
import math

df=pd.read_csv('iris.csv',header=None)
iris=df.iloc[0:100].values
dataset = df.iloc[0:100,[0,1,2,3]].values #dataset x1 sampai x4

weights=[0.5,0.5,0.5,0.5] #weight1-4
bias=[0.5]

def activation(row, t):
    result=0
    activation=0
    for i in range(len(row)):
        result += weights[i] * row[i]
    result=result+bias
    activation = 1/(1+math.exp(-result)) #output
    delta(row,activation,t) #delta weight dan delta bias
    return activation

def delta(row,activation,t):
    alfa=0.1
    for i in range(len(row)):
        d_weight=2*row[i]*(activation-t)*(1-activation)*activation #delta weight
        weights[i]=weights[i]-alfa*d_weight
        bias[0]=bias[0]-alfa*2*(activation-t)*(1-activation)*activation #bias berikutnya

def predict(activation):
    return 1.0 if activation >= 0.5 else 0.0

for row in dataset:
    if row[-1]>=1:
        t=1
    else:
        t=0
    output = activation(row, t)
    prediction=predict(output)
    error=math.pow((t-output),2)
    print("t=%d, prediction=%d, error=%s" % (t, prediction,error))