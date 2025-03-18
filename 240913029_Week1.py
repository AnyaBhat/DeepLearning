import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import correlate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import torch.nn as nn
from sympy.codegen.fnodes import reshape

"""
print("NUMPY\n")
arr1d=np.array([1,2,3,4,5])
print("1D Array:",arr1d)

arr2d=np.array([[1,2,3],[4,5,6]])
print("2D Array:",arr2d)

print("\nArray Operations:")
print("Sum Operations:",arr1d.sum())
print("Mean Operation;",arr1d.mean())
print("Standard Deviation:",arr1d.std())

reshaped_arr=arr1d.reshape(5,1)
print("\n Reshaped Array:\n",reshaped_arr)


print("\nPANDAS\n")
data={
    'Name': ['John','Emma','Alex','Sarah'],
    'Age':[28,24,32,27],
    'Salary':[50000,45000,65000,55000],
    'Department':['IT','HR','IT','Finance']
}

df=pd.DataFrame(data)
print("Sample DataFrame:\n",df)

print('\nDataFrame Info:')
print(df.info())

print("\nDataFrame Description")
print(df.describe())


print("\nMATPLOTLIB\n")

#Line Plot
plt.subplot(1,3,1)
x=np.linspace(0,10,100)

y=np.sin(x)

plt.plot(x,y)
plt.title('Line Plot:Sin Wave')
plt.xlabel('x')
plt.ylabel('sin(x)')

#Scatter Plot
plt.subplot(1,3,2)
x=np.random.normal(0,1,100)
print(x)
y=np.random.normal(0,1,100)
print(y)
plt.scatter(x,y)
plt.title('Scatter Plot')
plt.xlabel('x')
plt.ylabel('y')

#Histogram
plt.subplot(1,3,3)
data=np.random.normal(0,1,1000)
plt.hist(data,bins=30)
plt.title("Histogram")
plt.xlabel('Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


#Box Plot
plt.subplot(2,2,4)
sns.boxplot(x='Department',y='Salary',data=df)
plt.title('Salary Distribution by Department')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#Correlation Heatmap
plt.figure(figsize=(10,8))
numeric_cols=df.select_dtypes(include=[np.number]).columns
correlation=df[numeric_cols].corr()
sns.heatmap(correlation,annot=True,cmap='coolwarm',center=0)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()
"""



numpy_array= np.array([[1,2,3],[4,5,6]])
tensor_from_numpy=torch.from_numpy(numpy_array)
print("Tensor from Numpy",tensor_from_numpy)

tensor1=torch.tensor([[1,2,3],[4,5,6]])
print("\nDirect tensor",tensor1)

tensor2=torch.zeros((2,3))
tensor3=torch.ones((2,3))
tensor4=torch.randn((2,3))
tensor5=torch.arange(0,10,2)

print("Zero tensor",tensor2)
print("Ones tensor",tensor3)
print("Random tensor",tensor4)
print("Arange tensor",tensor5)



print("Tensor Operations")
print("Sum",tensor1.sum())
print("Mean",tensor1.float().mean())
print("Addition",tensor1+tensor1)
print("Multiplication",tensor1*tensor1)
print("Matrix Multiplication",torch.mm(tensor1,tensor1.T))

print("Reshape",tensor1,reshape(3,2))
print("Transpose",tensor1.T)
print("Concatenation",torch.cat([tensor1,tensor1],dim=0))
print("Stack",torch.stack([tensor1,tensor1]))

