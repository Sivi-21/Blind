# import cv2
# import numpy as np
# import pyttsx3

# # Initialize text-to-speech engine
# engine = pyttsx3.init()
# engine.setProperty('rate', 150)

# # Load YOLOv3 model
# net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# # Load class labels
# with open('coco.names', 'r') as f:
#     classes = [line.strip() for line in f.readlines()]

# # Get output layer names
# layer_names = net.getLayerNames()
# output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# # Start webcam
# cap = cv2.VideoCapture(0)

# # Detection thresholds
# confidence_threshold = 0.5
# nms_threshold = 0.4

# while True:
#     ret, frame = cap.read()

#     if not ret:
#         print("Error: Failed to capture the frame")
#         break

#     height, width = frame.shape[:2]
#     blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#     net.setInput(blob)
#     detections = net.forward(output_layers)

#     boxes = []
#     confidences = []
#     class_ids = []

#     for output in detections:
#         for detection in output:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]

#             if confidence > confidence_threshold:
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)

#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)

#     indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

#     detected_labels = set()

#     if len(indices) > 0:
#         for i in indices.flatten():
#             x, y, w, h = boxes[i]
#             label = str(classes[class_ids[i]])
#             confidence = confidences[i]

#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(frame, f"{label} {round(confidence, 2)}", (x, y - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#             detected_labels.add(label)

#         # Speak and exit after first detection
#         for label in detected_labels:
#             engine.say(label)
#             engine.runAndWait()
#         break  # Exit loop after detection

#     cv2.imshow('Webcam Object Detection', frame)

#     if cv2.waitKey(1) & 0xFF == ord('x'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()
# import cv2
# import numpy as np
# import pyttsx3
# import time

# # Initialize TTS
# engine = pyttsx3.init()
# engine.setProperty('rate', 150)

# # Load YOLO model
# net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
# with open('coco.names', 'r') as f:
#     classes = [line.strip() for line in f.readlines()]

# layer_names = net.getLayerNames()
# output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# # Start webcam
# cap = cv2.VideoCapture(0)

# confidence_threshold = 0.5
# nms_threshold = 0.4

# # Object tracking memory
# last_seen_objects = set()
# last_speech_time = 0
# speech_delay = 2  # seconds between speech updates

# print("Press 'x' to exit")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Camera error")
#         break

#     height, width = frame.shape[:2]
#     blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#     net.setInput(blob)
#     detections = net.forward(output_layers)

#     boxes, confidences, class_ids = [], [], []
#     for output in detections:
#         for detection in output:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > confidence_threshold:
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)

#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)

#     indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

#     current_objects = set()
#     if len(indices) > 0:
#         for i in indices.flatten():
#             x, y, w, h = boxes[i]
#             label = str(classes[class_ids[i]])
#             confidence = confidences[i]
#             current_objects.add(label)
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(frame, f"{label} {round(confidence, 2)}", (x, y - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#     # Speak if new objects appear (or scene changes)
#     now = time.time()
#     if (current_objects != last_seen_objects) and (now - last_speech_time > speech_delay):
#         if current_objects:
#             to_say = "I see " + ", ".join(current_objects)
#             print("Speaking:", to_say)
#             engine.say(to_say)
#             engine.runAndWait()
#         last_seen_objects = current_objects.copy()
#         last_speech_time = now

#     cv2.imshow("Object Detection", frame)

#     if cv2.waitKey(1) & 0xFF == ord('x'):
#         break

# cap.release()
# cv2.destroyAllWindows()
# import cv2
# import numpy as np
# import pyttsx3
# import time
# import os

# # Initialize TTS
# engine = pyttsx3.init()
# engine.setProperty('rate', 150)

# # Load YOLO model
# net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
# with open('coco.names', 'r') as f:
#     classes = [line.strip() for line in f.readlines()]

# layer_names = net.getLayerNames()
# output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# # Create output folder for frames
# output_folder = "saved_frames"
# os.makedirs(output_folder, exist_ok=True)

# # Start webcam
# cap = cv2.VideoCapture(0)

# confidence_threshold = 0.5
# nms_threshold = 0.4

# # Object tracking memory
# last_seen_objects = set()
# last_speech_time = 0
# speech_delay = 2  # seconds between speech updates

# # Timer setup
# start_time = time.time()
# run_duration = 3 * 60  # Run for 3 minutes

# frame_count = 0

# print("Press 'x' to exit or wait for timer to end")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Camera error")
#         break

#     height, width = frame.shape[:2]
#     blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#     net.setInput(blob)
#     detections = net.forward(output_layers)

#     boxes, confidences, class_ids = [], [], []
#     for output in detections:
#         for detection in output:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > confidence_threshold:
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)

#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)

#     indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

#     current_objects = set()
#     if len(indices) > 0:
#         for i in indices.flatten():
#             x, y, w, h = boxes[i]
#             label = str(classes[class_ids[i]])
#             confidence = confidences[i]
#             current_objects.add(label)
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(frame, f"{label} {round(confidence, 2)}", (x, y - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#     now = time.time()

#     # Speak and save frame if new objects appear
#     if (current_objects != last_seen_objects) and (now - last_speech_time > speech_delay):
#         if current_objects:
#             to_say = "I see " + ", ".join(current_objects)
#             print("Speaking:", to_say)
#             engine.say(to_say)
#             engine.runAndWait()

#             # Save frame
#             filename = os.path.join(output_folder, f"detected_{frame_count}.jpg")
#             cv2.imwrite(filename, frame)
#             print(f"Saved frame: {filename}")
#             frame_count += 1

#         last_seen_objects = current_objects.copy()
#         last_speech_time = now

#     cv2.imshow("Object Detection", frame)

#     if cv2.waitKey(1) & 0xFF == ord('x'):
#         print("Manual exit")
#         break

#     if now - start_time > run_duration:
#         print("Time limit reached. Exiting...")
#         break

# cap.release()
# cv2.destroyAllWindows()
import cv2
import numpy as np
import pyttsx3
import time
import os

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speech speed
engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)

# Load YOLO Object Detection Model
print("Loading YOLO model...")
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Load class names
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Get YOLO output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Create output folder for saving frames
output_folder = "saved_frames"
os.makedirs(output_folder, exist_ok=True)

# Initialize webcam
print("Starting webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open camera!")
    exit()

# Detection settings
confidence_threshold = 0.5  # Minimum confidence for detection
nms_threshold = 0.4  # Non-maximum suppression threshold

# Flag to track if we've already spoken
has_spoken = False

print("\n" + "="*50)
print("SINGLE DETECTION WITH AUTO-STOP")
print("="*50)
print("Will detect objects once and automatically exit")
print("="*50 + "\n")

# Give camera time to warm up
time.sleep(2)

# Main detection loop
while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Cannot read frame from camera")
        break

    height, width = frame.shape[:2]

    # Prepare image for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    
    # Run detection
    detections = net.forward(output_layers)

    # Process detections
    boxes = []
    confidences = []
    class_ids = []

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > confidence_threshold:
                # Get bounding box coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    # Collect detected objects
    current_objects = set()
    
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            current_objects.add(label)
            
            # Draw bounding box and label on frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {round(confidence, 2)}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display info on frame
    cv2.putText(frame, f"Objects Detected: {len(current_objects)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Show frame
    cv2.imshow("Object Detection - Will auto-close after detection", frame)

    # SPEAK ONCE AND EXIT
    if not has_spoken and current_objects:
        # Count objects
        object_list = list(current_objects)
        
        # Create speech text
        if len(object_list) == 1:
            to_say = f"I see a {object_list[0]}"
        elif len(object_list) == 2:
            to_say = f"I see a {object_list[0]} and a {object_list[1]}"
        else:
            to_say = "I see " + ", ".join(object_list[:-1]) + f", and a {object_list[-1]}"
        
        print(f"\n[VOICE] {to_say}")
        engine.say(to_say)
        engine.runAndWait()

        # Save frame with detections
        filename = os.path.join(output_folder, f"detected_final.jpg")
        cv2.imwrite(filename, frame)
        print(f"[SAVED] {filename}")
        
        # Mark as spoken
        has_spoken = True
        
        # Wait a moment to show the frame, then exit
        print("\nDetection complete. Exiting in 2 seconds...")
        cv2.waitKey(2000)  # Wait 2 seconds
        break

    # Allow manual exit with 'x' key
    if cv2.waitKey(1) & 0xFF == ord('x'):
        print("\n[EXIT] Manual exit by user")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

print("\n" + "="*50)
print(f"Session Complete!")
print(f"Frame saved in: {output_folder}/")
print("="*50)