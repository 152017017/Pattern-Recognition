import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("train.txt", header=None, sep=",")
data = df.values



xc1 = []
yc1 = []
xc2 = []
yc2 = []
class1 = []
class2 = []

for i in data:
    temp = i

    if (i[2] == 1):
        class1.append((temp[0], temp[1]))
    else:
        class2.append((temp[0], temp[1]))

plt.title('Class Values')
plt.xlabel('x-axis', color='black')
plt.ylabel('y-axis', color='black')

for a, b in class1:
    xc1.append(a)
    yc1.append(b)

for a, b in class2:
    xc2.append(a)
    yc2.append(b)

plt.scatter(xc1, yc1, color='red', marker='^', label='Class - 1 (Test Data)')
plt.scatter(xc2, yc2, color='blue', marker='*', label='Class - 2 (Test Data)')

plt.legend()
plt.show()



df = pd.read_csv("test.txt", header=None, sep=",")
data2 = df.values

k = 3

val = []
classify = []

            
for i in data2:
    temp = i
    X1 = temp[0]
    X2 = temp[1]
    
    val = []
    
    for j in data:
       temp2 = j     
       x1 = temp2[0]
       x2 = temp2[1]
       c = temp2[2]
       
       dist = ((x1-X1) * (x1-X1)) + ((x2-X2) * (x2-X2))
       val.append((dist,c))
    
    
    val = np.array(val)
    sort_val = val[val[:,0].argsort()]
     
    print('\nTest Point '+str(X1)+', '+str(X2))
    
    cnt = 1
    for m in sort_val:
        t = m
        
        if(cnt <=k):
            print('Distance '+str(cnt)+': '+str(t[0])+'\t'+'Class: '+str(t[1]))
            
        cnt = cnt+1
    
    
    
    count = 0
    cs1=0
    cs2=0
    for  p in sort_val:
        temp3 = p        
        if(count <=k-1):
            if(temp3[1] == 1):
                cs1 = cs1+1
            else:
                cs2 = cs2+1
     
        count = count+1
        
        
    if(cs1>cs2):
        classify.append((X1, X2, 1))
        print('Predicted Class: '+str(1))
    else:
        classify.append((X1, X2, 2))
        print('Predicted Class: '+str(2))

classify = np.array(classify)       
    
    



xxc1 = []
yyc1 = []
xxc2 = []
yyc2 = []
mclass1 = []
mclass2 = []

for i in classify:
    temp4 = i

    if (i[2] == 1):
        mclass1.append((temp4[0], temp4[1]))
    else:
        mclass2.append((temp4[0], temp4[1]))

plt.title('KNN - Class Values')
plt.xlabel('x-axis', color='black')
plt.ylabel('y-axis', color='black')

for a, b in mclass1:
    xxc1.append(a)
    yyc1.append(b)

for a, b in mclass2:
    xxc2.append(a)
    yyc2.append(b)

plt.scatter(xxc1, yyc1, color='k', marker='X', label='Class - 1 (Train Data-n)')
plt.scatter(xxc2, yyc2, color='y', marker='D', label='Class - 2 (Train Data-n)')

plt.legend()
plt.show()



