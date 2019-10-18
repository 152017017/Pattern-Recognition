import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("train.txt", header=None, sep=" ", dtype='float64')
data = df.values

xc1 = []
yc1 = []
xc2 = []
yc2 = []
class1 = []
class2 = []

for i in data:
    temp = i

    if (i[2] == 1.0):
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

################## part-1 End ##########################


total_y = []

for a, b in class1:
    total_y.append(a ** 2)
    total_y.append(b ** 2)
    total_y.append(a * b)
    total_y.append(a)
    total_y.append(b)
    total_y.append(1)

for a, b in class2:
    total_y.append((-1) * a ** 2)
    total_y.append((-1) * b ** 2)
    total_y.append((-1) * a * b)
    total_y.append((-1) * a)
    total_y.append((-1) * b)
    total_y.append((-1) * 1)

ss_array = np.zeros((6, 6))

count = 0
for x in range(6):
    for y in range(6):
        ss_array[x][y] = float(total_y[count])
        count = count + 1

################## part-2 End ##########################



#############  Batch Processing / Weight Vector = 0 #################

alpha = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])


# s_array = np.array([[0,0,1], [-1,1,-1], [-1,-1,-1]])

batch_process_w_zero = np.zeros(10)

for z in range(10):

    al = alpha[z]

    aSize = 6
    wT = np.zeros(aSize)
    yDemo = np.zeros(aSize)
    gY = np.zeros(aSize);
    mis = np.zeros(aSize);
    gY_check = 0;
    iteration = 0;
    temp = 0;

    while (gY_check != aSize):
        gY_check = 0
        
        for x in range(aSize):
            for y in range(aSize):
                yDemo[y] = ss_array[x][y];

            gY[x] = np.dot(yDemo, wT)

            if (gY[x] <= 0):
                mis = mis + yDemo;
                temp = 1;
            else:
                gY_check = gY_check + 1

        iteration = iteration + 1


        mis = mis * al

        if (temp == 1):
            wT = mis + wT
        else:
            wT = (al) * ss_array.sum(axis=0)

       # print("\n")
        mis = np.zeros(aSize)
        temp = 0

    #print("total iteration: " + str(iteration))
    batch_process_w_zero[z] = iteration
    #print("alpha: "+str(alpha[z]))
    #print(iteration)
    #print("\n")



#############  Single Processing / Weight Vector = 0  #################


alpha = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
single_process_w_zero = np.zeros(10)

for zz in range(10):
    al = alpha[zz]

    aSize = 6
    wT = np.zeros(aSize)
    yDemo = np.zeros(aSize)
    gY = np.zeros(aSize);
    mis = np.zeros(aSize);
    update_factor = np.zeros(aSize)
    gY_check = 0;
    iteration = 0;
    temp = 0;
    cnt = 0;

    # s_array = np.array([[0,0,1], [-1,-1,-1], [-1,1,-1]])

    while (gY_check != aSize):
        gY_check = 0

        for x in range(aSize):
            for y in range(aSize):
                yDemo[y] = ss_array[x][y];

            gY[x] = np.dot(yDemo, wT)

            if (gY[x] <= 0):
                update_factor = al * yDemo;
                wT = wT + update_factor
            else:
                update_factor = np.zeros(aSize)
                gY_check = gY_check + 1

        iteration = iteration + 1


        #print("\n")
        update_factor = np.zeros(aSize)

    single_process_w_zero[zz] = iteration
   # print("alpha: " + str(alpha[zz]))
   # print(iteration)



#############  Batch Processing / Weight Vector = 1 #################

alpha = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])


# s_array = np.array([[0,0,1], [-1,1,-1], [-1,-1,-1]])

batch_process_w_one = np.zeros(10)

for z in range(10):

    al = alpha[z]

    aSize = 6
    wT = np.ones(aSize)
    yDemo = np.zeros(aSize)
    gY = np.zeros(aSize);
    mis = np.zeros(aSize);
    gY_check = 0;
    iteration = 0;
    temp = 0;

    while (gY_check != aSize):
        gY_check = 0
        
        for x in range(aSize):
            for y in range(aSize):
                yDemo[y] = ss_array[x][y];

            gY[x] = np.dot(yDemo, wT)

            if (gY[x] <= 0):
                mis = mis + yDemo;
                temp = 1;
            else:
                gY_check = gY_check + 1

        iteration = iteration + 1
        

        mis = mis * al

        if (temp == 1):
            wT = mis + wT
        else:
            wT = (al) * ss_array.sum(axis=0)

       # print("\n")
        mis = np.zeros(aSize)
        temp = 0

    #print("total iteration: " + str(iteration))
    batch_process_w_one[z] = iteration
    #print("alpha: "+str(alpha[z]))
   # print(iteration)
    #print("\n")



#############  Single Processing / Weight Vector = 1 #################


alpha = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
single_process_w_one = np.zeros(10)

for zz in range(10):
    al = alpha[zz]

    aSize = 6
    wT = np.ones(aSize)
    yDemo = np.zeros(aSize)
    gY = np.zeros(aSize);
    mis = np.zeros(aSize);
    update_factor = np.zeros(aSize)
    gY_check = 0;
    iteration = 0;
    temp = 0;
    cnt = 0;

    # s_array = np.array([[0,0,1], [-1,-1,-1], [-1,1,-1]])

    while (gY_check != aSize):
        gY_check = 0

        for x in range(aSize):
            for y in range(aSize):
                yDemo[y] = ss_array[x][y];

            gY[x] = np.dot(yDemo, wT)

            if (gY[x] <= 0):
                update_factor = al * yDemo;
                wT = wT + update_factor
            else:
                update_factor = np.zeros(aSize)
                gY_check = gY_check + 1

        iteration = iteration + 1

        #print("\n")
        update_factor = np.zeros(aSize)

    single_process_w_one[zz] = iteration
    #print("alpha: " + str(alpha[zz]))
    #print(iteration)
    


#############  Batch Processing / Weight Vector = random #################

alpha = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])


# s_array = np.array([[0,0,1], [-1,1,-1], [-1,-1,-1]])

batch_process_w_random = np.zeros(10)

for z in range(10):

    al = alpha[z]

    aSize = 6
    wT = np.random.rand(aSize)
    yDemo = np.zeros(aSize)
    gY = np.zeros(aSize);
    mis = np.zeros(aSize);
    gY_check = 0;
    iteration = 0;
    temp = 0;

    while (gY_check != aSize):
        gY_check = 0
        
        for x in range(aSize):
            for y in range(aSize):
                yDemo[y] = ss_array[x][y];

            gY[x] = np.dot(yDemo, wT)

            if (gY[x] <= 0):
                mis = mis + yDemo;
                temp = 1;
            else:
                gY_check = gY_check + 1

        iteration = iteration + 1
        

        mis = mis * al

        if (temp == 1):
            wT = mis + wT
        else:
            wT = (al) * ss_array.sum(axis=0)

       # print("\n")
        mis = np.zeros(aSize)
        temp = 0

    #print("total iteration: " + str(iteration))
    batch_process_w_random[z] = iteration
   # print("alpha: "+str(alpha[z]))
    #print(iteration)
    #print("\n")



#############  Single Processing / Weight Vector = random #################


alpha = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
single_process_w_random = np.zeros(10)

for zz in range(10):
    al = alpha[zz]

    aSize = 6
    wT = np.random.rand(aSize)
    yDemo = np.zeros(aSize)
    gY = np.zeros(aSize);
    mis = np.zeros(aSize);
    update_factor = np.zeros(aSize)
    gY_check = 0;
    iteration = 0;
    temp = 0;
    cnt = 0;

    # s_array = np.array([[0,0,1], [-1,-1,-1], [-1,1,-1]])

    while (gY_check != aSize):
        gY_check = 0

        for x in range(aSize):
            for y in range(aSize):
                yDemo[y] = ss_array[x][y];

            gY[x] = np.dot(yDemo, wT)

            if (gY[x] <= 0):
                update_factor = al * yDemo;
                wT = wT + update_factor
            else:
                update_factor = np.zeros(aSize)
                gY_check = gY_check + 1

        iteration = iteration + 1

        #print("\n")
        update_factor = np.zeros(aSize)

    single_process_w_random[zz] = iteration
    #print("alpha: " + str(alpha[zz]))
    #print(iteration)
    


print("Initial Weight Vector All One: ")
print("Value of Alpha"+" \t" + "One at a Time" + "\t"+ "Many at a Time")
for x in range(10):
    print(str(alpha[x])+"\t\t"+str(single_process_w_one[x])+"\t\t" +str(batch_process_w_one[x]))
    
print("Initial Weight Vector All Zero: ")
print("Value of Alpha"+" \t" + "One at a Time" + "\t"+ "Many at a Time")
for x in range(10):
    print(str(alpha[x])+"\t\t"+str(single_process_w_zero[x])+"\t\t" +str(batch_process_w_zero[x]))


print("Initial Weight Vector All Random: ")
print("Value of Alpha"+" \t" + "One at a Time" + "\t"+ "Many at a Time")
for x in range(10):
    print(str(alpha[x])+"\t\t"+str(single_process_w_random[x])+"\t\t" +str(batch_process_w_random[x]))



n_groups = 10

fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.2
opacity = 0.8

plt.bar(index, single_process_w_zero, bar_width,
alpha=opacity,
color='b',
label='one at a time')

plt.bar(index + bar_width, batch_process_w_zero, bar_width,
alpha=opacity,
color='r',
label='many at a time')

plt.xlabel('Learning Rate')
plt.ylabel('Number of Iteration')
plt.title('Initial Weight Vector All Zero')
plt.xticks(index + bar_width, ('0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9','1'))
plt.legend()

plt.tight_layout()
plt.show()

#fig.savefig('Initial Weight Vector All Zero.png')
#plt.close(fig)

n_groups = 10

fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.2
opacity = 0.8

plt.bar(index, single_process_w_one, bar_width,
alpha=opacity,
color='b',
label='one at a time')

plt.bar(index + bar_width, batch_process_w_one, bar_width,
alpha=opacity,
color='r',
label='many at a time')

plt.xlabel('Learning Rate')
plt.ylabel('Number of Iteration')
plt.title('Initial Weight Vector All One')
plt.xticks(index + bar_width, ('0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9','1'))
plt.legend()

plt.tight_layout()
plt.show()

#fig.savefig('Initial Weight Vector All One.png')
#plt.close(fig)


n_groups = 10

fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.2
opacity = 0.8

plt.bar(index, single_process_w_random, bar_width,
alpha=opacity,
color='b',
label='one at a time')

plt.bar(index + bar_width, batch_process_w_random, bar_width,
alpha=opacity,
color='r',
label='many at a time')

plt.xlabel('Learning Rate')
plt.ylabel('Number of Iteration')
plt.title('Initial Weight Vector All Random')
plt.xticks(index + bar_width, ('0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9','1'))
plt.legend()

plt.tight_layout()
plt.show()

#fig.savefig('Initial Weight Vector All Random.png')
#plt.close(fig)
