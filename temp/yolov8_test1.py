from ultralytics import YOLO
import cv2

model = YOLO('yolov8m-pose.pt')

cap = cv2.VideoCapture(0)  # Open the default camera (source=0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.3)  # Perform inference
    annotated_frame = results[0].plot()  # Get the annotated frame

    cv2.imshow('YOLOv8 Pose Estimation', annotated_frame)
    cv2.imwrite('results.jpg', annotated_frame)  # Save the frame

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Check for 'q' key press
        break

cap.release()
cv2.destroyAllWindows()

