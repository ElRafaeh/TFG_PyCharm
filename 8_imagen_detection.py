import cv2
from utils import Algorithm, Image
from matplotlib import pyplot as plt

img = cv2.imread('images/mayor.png')
# cap = cv2.VideoCapture('datasets/MCFD/chute01/cam3.avi')

algoritmo = Algorithm()


fall = algoritmo.run(img)

cv2.putText(img, "FALL: "+str(fall), (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), thickness=4)

img2 = Image.from_file('images/mayor.png')


f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.imshow(img2.image)
ax1.get_xaxis().set_visible(False)
ax1.get_yaxis().set_visible(False)


l2 = ax2.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax2.get_xaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)


plt.show()