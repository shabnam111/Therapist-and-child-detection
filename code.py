#pip install opencv-python torch yolov8 deep_sort_pytorch
import cv2
import torch
from yolov8 import YOLOv8  
from deep_sort.deep_sort import DeepSort

# Initialize YOLOv8 and DeepSort
yolo_model = YOLOv8(weights='yolov8x.pt', classes=['child', 'therapist'])
deep_sort_tracker = DeepSort(model_path='ckpt.t7')

def process_video(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = yolo_model.predict(frame)

        
        bbox_xywh = []
        confidences = []
        class_ids = []
        for detection in detections:
            bbox_xywh.append(detection[:4])
            confidences.append(detection[4])
            class_ids.append(detection[5])

        
        outputs = deep_sort_tracker.update(bbox_xywh, confidences, class_ids, frame)

       
        for output in outputs:
            x1, y1, x2, y2, obj_id, cls_id = output
            label = f'{yolo_model.classes[cls_id]} {obj_id}'
            color = (0, 255, 0) if cls_id == 0 else (0, 0, 255)  
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        out.write(frame)

    cap.release()
    out.release()

#youtube-dl -f best -o video.mp4 https://www.youtube.com/watch?v=V9YDDpo9LWg

input_video = 'path_to_your_input_video.mp4'
output_video = 'path_to_your_output_video.mp4'
process_video(input_video, output_video)
