import matplotlib.pyplot as plt
import numpy as np

data = []

x = []
y = []
cls1 = []
cls2 = []

sum_x1 = 0; sum_x2 = 0
sum_y1 = 0; sum_y2 = 0

with open('train.txt', 'r') as f:
    for line in f:
        data = line.split()
        x.append(int(data[0]))
        y.append(int(data[1]))

        xx = int(data[0])
        yy = int(data[1])

        if int(data[2]) == 1:
            cls1.append((xx, yy))
            sum_x1 = sum_x1 + xx
            sum_y1 = sum_y1 + yy

        else:
            cls2.append((xx, yy))
            sum_x2 = sum_x2 + xx
            sum_y2 = sum_y2 + yy

mean_x1 = sum_x1 // len(cls1)
mean_y1 = sum_y1 // len(cls1)
mean_x2 = sum_x2 // len(cls2)
mean_y2 = sum_y2 // len(cls2)

cls1_mean = [mean_x1, mean_y1]
cls2_mean = [mean_x2, mean_y2]

min_x = min(x)
max_x = max(x)

min_y = min(y)
max_y = max(y)

plt.figure(figsize=(max_x*2, max_y*2))
plt.title('K_MEAN Classifier')
plt.xlabel('x-axis', color='red')
plt.ylabel('y-axis', color='red')

# plotting values of class - 1
label_cnt = 0

for a, b in cls1:
    if label_cnt is 0:
        plt.scatter(a, b, color='red', marker='^', label='Class - 1 (Train Data).')
        label_cnt = label_cnt + 1
    plt.scatter(a, b, color='red', marker='^')

# plotting values of class - 2
label_cnt = 0

for a, b in cls2:
    if label_cnt is 0:
        plt.scatter(a, b, color='green', marker='*', label='Class - 2 (Train Data).')
        label_cnt = label_cnt + 1
    plt.scatter(a, b, color='green', marker='*')

data.clear()

cls1_cnt = 0; cls2_cnt = 0
cnt = 0
total = 0

with open('test.txt', 'r') as f1:
    for line in f1:
        data = line.split()
        total = total + 1

        xx = int(data[0])
        yy = int(data[1])
        cl = int(data[2])

        cls = np.array([xx, yy])

        gx1 = np.dot(cls1_mean, cls) - (0.5 * np.dot(cls1_mean, cls1_mean))
        gx2 = np.dot(cls2_mean, cls) - (0.5 * np.dot(cls2_mean, cls2_mean))

        if gx1 >= gx2:
            if cl is 1:
                cnt = cnt+1

            cls1_cnt = cls1_cnt + 1
            if cls1_cnt is 1:
                plt.scatter(xx, yy, color='red', marker='s', label='Class - 1 (Test Data).')
            else:
                plt.scatter(xx, yy, color='red', marker='s')

        else:
            if cl is 2:
                cnt = cnt+1

            cls2_cnt = cls2_cnt + 1
            if cls2_cnt is 1:
                plt.scatter(xx, yy, color='green', marker='p', label='Class - 2 (Test Data).')
            else:
                plt.scatter(xx, yy, color='green', marker='p')


accuracy_cls = (cnt / total) * 100

print('cnt: ', cnt)
print('total: ', total)
print('Accuracy cls : ', accuracy_cls, '%')

m1 = mean_x1 - mean_x2
m2 = mean_y1 - mean_y2

cons = -0.5*(np.dot(cls1_mean, cls1_mean) - np.dot(cls2_mean, cls2_mean))

x = np.arange(min_x * 2, max_x * 2, 0.1)
y = (m1*x + cons) / (-m2)

print('x: ', x)
print('y: ', y)

plt.plot(x, y, label='Decision Boundary', color='purple')
plt.legend()
plt.show()
