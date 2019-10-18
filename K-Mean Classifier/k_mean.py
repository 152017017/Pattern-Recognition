import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import math

df = pd.read_csv("data.txt", header=None, sep=" ")
data = df.values


v1 = []
v2 = []

for x in data:
    v1.append(x[0])
    v2.append(x[1])

v1= np.array(v1)
v2 = np.array(v2)

plt.title('Class Values')
plt.xlabel('x-axis', color='black')
plt.ylabel('y-axis', color='blue')
plt.scatter(v1, v2, color='blue', marker='*', label='Test Data')
plt.legend()
plt.show()


k = 2
init = 1
while(True):
    if(init == 1):
        m1_index = random.randint(1,len(data))
        m2_index = random.randint(1,len(data))
        m1 = data[m1_index]
        m2 = data[m2_index]
           
    
        temp_data = np.zeros((len(data),2))
        cnt = 0
        for x in data:
            x1 = x[0]
            y1 = x[1]
            xm1 = m1[0]
            ym1 = m1[1] 
            xm2 = m2[0]
            ym2 = m2[1]
        
            dm1 = math.sqrt( ((x1-xm1)**2) + ((y1-ym1)**2) )
            dm2 = math.sqrt( ((x1-xm2)**2) + ((y1-ym2)**2) )
            
            temp_data[cnt][0] = dm1
            temp_data[cnt][1] = dm2
        
            cnt = cnt+1
        
        cnt = 0
        clstr1 = []
        clstr2 = []
        for x in temp_data:
            if(x[0] <= x[1]):
                clstr1.append(cnt+1)
            else:
                clstr2.append(cnt+1)
            cnt = cnt + 1   
                          
        init = init + 1
            
        
    else:             
        m1 = []
        m2 = []
        mmm1 = 0.0       
        for p in range( len(clstr1) ):           
            mmm1 = mmm1 + data[ clstr1[p]-1 ][0]
        mmm1 = mmm1 / len(clstr1)
        
        mmm2 = 0.0       
        for p in range( len(clstr1) ):           
            mmm2 = mmm2 + data[ clstr1[p]-1 ][1]
        mmm2 = mmm2 / len(clstr1)
            
           
        m1.append(mmm1)
        m1.append(mmm2)
                      
       
        mmm1 = 0.0       
        for p in range( len(clstr2) ):           
            mmm1 = mmm1 + data[ clstr2[p]-1 ][0]
        mmm1 = mmm1 / len(clstr2)
        
        mmm2 = 0.0       
        for p in range( len(clstr2) ):           
            mmm2 = mmm2 + data[ clstr2[p]-1 ][1]
        mmm2 = mmm2 / len(clstr2)
            
           
        m2.append(mmm1)
        m2.append(mmm2)               
        m1 = np.array(m1)
        m2 = np.array(m2)
                
        
        temp_data = np.zeros((len(data),2))
        cnt = 0
        for x in data:
            x1 = x[0]
            y1 = x[1]
            xm1 = m1[0]
            ym1 = m1[1] 
            xm2 = m2[0]
            ym2 = m2[1]
        
            dm1 = math.sqrt( ((x1-xm1)**2) + ((y1-ym1)**2) )
            dm2 = math.sqrt( ((x1-xm2)**2) + ((y1-ym2)**2) )
            
            temp_data[cnt][0] = dm1
            temp_data[cnt][1] = dm2
        
            cnt = cnt+1
        
        cnt = 0
        clstr11 = []
        clstr22 = []
        for x in temp_data:
            if(x[0] <= x[1]):
                clstr11.append(cnt+1)
            else:
                clstr22.append(cnt+1)
            cnt = cnt + 1                                    
        
        if(clstr1 == clstr11):
            break;
        else:
             clstr1 = clstr11
             clstr2 = clstr22           
        
c1x = []
c1y = []
c2x = []
c2y = []
for x in clstr1:
    c1x.append(data[x-1][0])    
    c1y.append(data[x-1][1])
for x in clstr2:
    c2x.append(data[x-1][0])    
    c2y.append(data[x-1][1])
       
        
plt.title('Class Values')
plt.xlabel('x-axis', color='black')
plt.ylabel('y-axis', color='black')
plt.scatter(c1x, c1y, color='green', marker='*', label='Cluster-1')
plt.scatter(c2x, c2y, color='red', marker='*', label='Cluster-2')
plt.legend()
plt.show()     