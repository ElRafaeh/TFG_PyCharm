from utils import MonocularMapper, Image
import matplotlib.pyplot as plt
import cv2

img = Image.from_file('images/mayor.png')

mapper = MonocularMapper(1)
depth_1, time1 = mapper.map(img.image, True)
depth1 = Image.from_array(depth_1)

mapper2 = MonocularMapper(2)
depth_2, time2 = mapper2.map(img.image, True)
depth2 = Image.from_array(depth_2)

mapper3 = MonocularMapper(3)
depth_3, time3 = mapper3.map(img.image, True)
depth3 = Image.from_array(depth_3)

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
ax1.imshow(depth1.image)
ax1.text(5, 5, round(time1, 3), bbox={'facecolor': 'white', 'pad': 10})
ax1.set_title('MiDaS Small Model')

l2 = ax2.imshow(depth2.image)
ax2.text(5, 5, round(time2, 3), bbox={'facecolor': 'white', 'pad': 10})
ax2.set_title('MiDaS DPT_Hybrid')
# ax2.legend(loc="upper right")

l3 = ax3.imshow(depth3.image)
ax3.text(5, 5, round(time3, 3), bbox={'facecolor': 'white', 'pad': 10})
ax3.set_title('MiDaS DPT_Large')
# ax3.legend(loc="upper right")

plt.show()

