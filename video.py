import cv2
from utils import PoseEstimator, Algorithm, MonocularMapper

cap = cv2.VideoCapture('datasets/FallDataset/Coffee_room_01/Videos/video (2).avi')
estimator = PoseEstimator('models/pose_landmarker_full.task')
mapper = MonocularMapper(2)
algoritmo = Algorithm(estimator, mapper)
count = 0

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