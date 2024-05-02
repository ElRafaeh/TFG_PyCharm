from utils import MonocularMapper, Image
import matplotlib.pyplot as plt

img = Image.from_file('images/mayor.png')

mapper = MonocularMapper(2)
depth_1 = mapper.map(img.image)
depth1 = Image.from_array(depth_1)


f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.imshow(img.image)
ax1.get_xaxis().set_visible(False)
ax1.get_yaxis().set_visible(False)


l2 = ax2.imshow(depth1.image)
ax2.get_xaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)


plt.show()
