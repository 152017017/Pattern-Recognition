import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import axes3d

#task 1
df = pd.read_csv("test.txt", sep=",", header=None, dtype="float")
row, col = df.shape

x_vals = np.array([[df[j][i] for j in range(col)] for i in range(row)])
u1 = np.array([0,0])
u2 = np.array([2,2])
sig1 = np.array([[0.25,0.3],[0.3,1]])
sig2 = np.array([[0.5,0],[0,0.5]])

def generatePDF(x_vals, vertex_count, f):
    const1 = 1/math.sqrt(pow(2*3.14,2)*np.linalg.det(sig1))
    const2 = 1/math.sqrt(pow(2*3.14,2)*np.linalg.det(sig2))

    n1 = np.array([const1 * np.exp((-0.5) * np.dot(np.dot(x_vals[i] - u1, np.linalg.inv(sig1)), x_vals[i] - u1)) for i in range(vertex_count)])
    n2 = np.array([const2 * np.exp((-0.5) * np.dot(np.dot(x_vals[i] - u2, np.linalg.inv(sig2)), x_vals[i] - u2)) for i in range(vertex_count)])

    if f==0:
        pdf_and_classes = np.array([[n1[i] if n1[i] > n2[i] else n2[i] for i in range(vertex_count)],[1 if n1[i] > n2[i] else 2 for i in range(vertex_count)]])
        return pdf_and_classes
    else:
        pdf = np.array([n1, n2])
        return pdf

pdf_and_classes = generatePDF(x_vals, row, 0)
pdf_vals = pdf_and_classes[0]
classes = pdf_and_classes[1]

class1 = np.transpose(np.array([x_vals[i] for i in range(row) if classes[i] == 1]))
class2 = np.transpose(np.array([x_vals[i] for i in range(row) if classes[i] == 2]))
class1_pdf = np.array([pdf_vals[i] for i in range(row) if classes[i] == 1])
class2_pdf = np.array([pdf_vals[i] for i in range(row) if classes[i] == 2])

#task 2,3,4
x1 = np.linspace(-6, 6, 121)
x2 = np.linspace(-6, 6, 121)
x1Grid, x2Grid = np.meshgrid(x1, x2)

graph_row, graph_col = x1Grid.shape
x_vals_graph = []
for i in range(graph_row):
    for j in range(graph_col):
        coords = []
        coords.append(x1Grid[i][j])
        coords.append(x2Grid[i][j])
        x_vals_graph.append(coords)
x_vals_graph = np.array(x_vals_graph)

vertex_count, _ = x_vals_graph.shape
pdf_vals = generatePDF(x_vals_graph, vertex_count, 1)
pdf_class1 = pdf_vals[0]
pdf_class2 = pdf_vals[1]
pdf_class1_graph_2d = pdf_class1.reshape(graph_row, graph_col)
pdf_class2_graph_2d = pdf_class2.reshape(graph_row, graph_col)
pdf_and_classes = generatePDF(x_vals_graph, vertex_count, 0)
pdf_class_graph_2d = pdf_and_classes[0]
pdf_class_graph_2d = pdf_class_graph_2d.reshape(graph_row, graph_col)

class1_count = class1[0].shape
z1_const = np.array([0 for i in range(class1_count[0])])
class2_count = class2[0].shape
z2_const = np.array([0 for i in range(class2_count[0])])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Probability Density')
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_zlim(-0.5, 0.5)
ax.plot_surface(x1Grid, x2Grid, pdf_class_graph_2d, cmap="viridis", rstride=1, cstride=1, alpha=0.8)
ax.contour(x1Grid, x2Grid, pdf_class1_graph_2d - pdf_class2_graph_2d, 10, cmap="viridis", offset=-0.5, linestyles="solid", alpha=0.5)
ax.scatter(class1[0], class1[1], z1_const, color='red', marker='o')
ax.scatter(class2[0], class2[1], z2_const, color='blue', marker='o')
plt.show()
