import cv2
from utils import Algorithm

cap = cv2.VideoCapture('datasets/imvia/Home_02/Videos/video (37).avi')
# cap = cv2.VideoCapture('datasets/MCFD/chute01/cam3.avi')

algoritmo = Algorithm(2)
count = 0

debug = False

#
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    fall = algoritmo.run(frame, debug)
    frame = cv2.cvtColor(algoritmo.image.image, cv2.COLOR_RGB2BGR)

    cv2.putText(frame, "FALL: "+str(fall), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
    
    for i, landmark in enumerate(algoritmo.landmarks):
            cv2.circle(frame, (int(landmark[0]), int(landmark[1])), 1, (255, 0, 0), 2)
            # cv2.scatter(int(landmark[0]), int(landmark[1]), c='#0CFF00', s=6)
            cv2.putText(frame, str(i), (int(landmark[0])+1, int(landmark[1])+1), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    else:
        debug = key == ord('d')

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()