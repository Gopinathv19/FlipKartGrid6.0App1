import mysql.connector
import streamlit as st
import cv2
import torch
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO
import time
import pandas as pd


# aws database configuration

def setup_database():
    conn = mysql.connector.connect(
        host="banana-rds.c9q0ke8ui71z.us-east-1.rds.amazonaws.com",   
        user="gopinathv19",          
        password="D62OakvF3mQCHNb4YJgZ",   
        database="banana_rds"        
    )
    return conn


 # values for the freshness data table inserting logic 

def save_to_database(freshness_data, conn, produce="banana"):
 
    cursor = conn.cursor()
    for freshness in freshness_data:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        freshness_text = {0: 'ripe', 1: 'rotten', 2: 'unripe'}[freshness]
        shelf_life_days = 7 if freshness == 0 else 0 if freshness == 1 else 14  # Days for ripe, rotten, unripe

        cursor.execute("""
            INSERT INTO FreshnessData (Timestamp, Produce, Freshness, Expected_Life_Span_Days)
            VALUES (%s, %s, %s, %s)
        """, (timestamp, produce, freshness_text, shelf_life_days))
    conn.commit()


# Fetching Data from the database
def fetch_from_database(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT Sl_no, Timestamp, Produce, Freshness, Expected_Life_Span_Days FROM FreshnessData")
    return cursor.fetchall()


def truncate_table(conn):
    cursor = conn.cursor()
    cursor.execute("TRUNCATE TABLE FreshnessData")
    conn.commit()

# yolo model Initialization

yolo_model = YOLO('best.pt')

#yolo classes

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'TV', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# sending the pridicted class

def classify_banana(image, model):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# Loading efficient net model for the classification

def load_efficientnet_model(checkpoint_path, num_classes=3):
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

efficientnet_model = load_efficientnet_model('bananaClassifier.pt', num_classes=3)

# Image Transformation Pipeline for classification
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Freshness labels
freshness_labels = {0: "ripe", 1: "rotten", 2: "unripe"}

# Streamlit App
st.title("Banana Freshness Detection (Live Video Feed)")

# Set up database connection
conn = setup_database()

if st.sidebar.button("Clear Table Data"):
    truncate_table(conn)
    st.sidebar.success("Table data cleared!") 

# Seting  up OpenCV for video capture
cap = cv2.VideoCapture(0)  # 0 for the default camera, adjust if you have multiple cameras

if not cap.isOpened():
    st.error("Could not open webcam.")
else:
    if 'run_detection' not in st.session_state:
        st.session_state.run_detection = False

    stframe = st.empty()  # Placeholder for video feed
    stop_button = st.button("Stop Detection")
    start_button = st.button("Start Detection")

    if stop_button:
        st.session_state.run_detection = False

    if start_button:
        st.session_state.run_detection = True

    if st.session_state.run_detection:
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            if not ret:
                st.error("Failed to capture image.")
                break

            # YOLO Object Detection
            results = yolo_model(frame, conf=0.3)
            detections = results[0].boxes
            detected_classes = detections.cls.cpu().numpy()

            banana_detections = []
            for i in range(len(detections)):
                xyxy = detections.xyxy[i].cpu().numpy()
                class_id = int(detected_classes[i])
                if COCO_CLASSES[class_id] == "banana":
                    banana_detections.append(xyxy)

            freshness_data = []
            for xyxy in banana_detections:
                x1, y1, x2, y2 = map(int, xyxy)
                banana_crop = frame[y1:y2, x1:x2]
                banana_crop_pil = Image.fromarray(cv2.cvtColor(banana_crop, cv2.COLOR_BGR2RGB))

                label = classify_banana(banana_crop_pil, efficientnet_model)
                freshness_data.append(label)

                label_text = freshness_labels[label]
                colors = {0: (0, 255, 0), 1: (255, 165, 0), 2: (255, 0, 0)}
                cv2.rectangle(frame, (x1, y1), (x2, y2), colors[label], 2)
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            save_to_database(freshness_data, conn)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB", use_container_width=True)

    else:
        st.write("Start Detection or  See database ")

    # Fetch and display database data
    st.sidebar.title("Database Contents")
    data = fetch_from_database(conn)
    df = pd.DataFrame(data, columns=["Sl_no", "Timestamp", "Produce", "Freshness", "Expected_Life_Span_Days"])
    st.sidebar.table(df.set_index('Sl_no', drop=True))

    cap.release()
    cv2.destroyAllWindows()

 
if conn.is_connected():
    conn.close()
