import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
#(x='FPS',y='mIoU')
#
# plt.figure(figsize=(6, 3))
# plt.plot(6, 3)
# plt.plot(3, 3 * 2)

plt.xlabel('FPS')
plt.ylabel('mIoU')
plt.axis([0,155,50,80])

x = [45.6,151.6,7.6,11.2,62.4,135.0,71.1,45.7,66.1,109.5]
y = [56.1,58.3,59.8,69.5,68.0,60.3,68.4,74.7,75.3,73.6]
plt.scatter(x,y)
plt.savefig('table.png')
