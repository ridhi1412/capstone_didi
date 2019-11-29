import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import numpy as np

# x = np.linspace(0, 10, 5000)
# y = ss.expon.pdf(x)

# plt.figure(1)
# plt.plot(x, y)
# plt.title('Plot of the Exponential Distribution')
# plt.xlabel('X')
# plt.ylabel('Probability Density')
# plt.show()
# plt.savefig('Exponential Distribution.png')

# plt.figure(2)
# threshold = 4
# x = np.linspace(0, 10, 5000)
# y = ss.expon.pdf(x-threshold)
#
# idx = np.where(y==0)[0]
# y[idx] = 1
# plt.plot(x, y)
# plt.title('Plot of the Shifted Exponential Distribution')
# plt.xlabel('X')
# plt.ylabel('Probability Density')
# # plt.show()
# plt.savefig('Shifted_Exponential_Distribution.png')


plt.figure(2)
x = [1, 2, 3, 4, 5]
y = [0.48, 0.66, 0.66, 0.73, 0.77]
val = ['Model ' + str(i) for i in x]

color = [(102/255, 153/255, 255/255), (102/255, 102/255, 255/255), (51/255, 51/255, 51/255),
         (51/255, 51/255, 51/255), (0/255, 0/255, 255/255)]

plt.bar(x,y)
plt.xticks(x, val)
plt.title('Model Performance')
# plt.xlabel('Model')
plt.ylabel('Score')
plt.savefig('Model_Performance.png')
