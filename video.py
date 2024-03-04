import cv2
from utils import PoseEstimator, Algorithm, MonocularMapper

cap = cv2.VideoCapture('CAIDAS.mp4')
estimator = PoseEstimator('models/pose_landmarker_full.task')
mapper = MonocularMapper(1)
algoritmo = Algorithm(estimator, mapper)
count = 0

# cap.set(cv2.CAP_PROP_POS_FRAMES, 25*7)
# ret, frame = cap.read()
# # cv2.imshow('frame', frame)
# # cv2.waitKey(0)
# algoritmo.run(frame, True)

debug = False

#
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    fall = algoritmo.run(frame, debug)

    cv2.putText(frame, "FALL: "+str(fall), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))

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