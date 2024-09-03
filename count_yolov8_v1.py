import cv2,json
from ultralytics import YOLO, solutions
# 9/2/2024  第一版可以計數 輸出變數out_count 輸出save_data.json , 只有測試pushup

model = YOLO("trained/yolov8n-pose.pt")

# cap = cv2.VideoCapture(0)  # Open the default camera (source=0)
cap = cv2.VideoCapture("video/pushup2.mp4")


assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("count_yolov8_v1.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

gym_object = solutions.AIGym(
    line_thickness=2,
    view_img=True,
    pose_type="pushup",
    kpts_to_check=[6, 8, 10],
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    results = model.track(im0, verbose=False)  # Tracking recommended
    # results = model.predict(im0)  # Prediction also supported
    im0 = gym_object.start_counting(im0, results)
    out_count = gym_object.count
    print(out_count)
    with open("json/save_data.json", "w") as f:
        json.dump(out_count, f)
    video_writer.write(im0)

cv2.destroyAllWindows()
video_writer.release()



