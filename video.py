import cv2
from utils import PoseEstimator, Algorithm, MonocularMapper

cap = cv2.VideoCapture('videos/video_caidas.mp4')
estimator = PoseEstimator('models/pose_landmarker_full.task')
mapper = MonocularMapper(3)
algoritmo = Algorithm(estimator, mapper)
count = 0

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    count += 30
    cap.set(cv2.CAP_PROP_POS_FRAMES, count)

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #
    # algoritmo.run(cv2.resize(frame, (480, 480)))

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()