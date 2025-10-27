from ultralytics import YOLO
import cv2, os
import cvzone
import math
import numpy as np
from werkzeug.utils import secure_filename
from utils import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos, get_device
from flask import Flask, render_template, Response, request, jsonify

# Load the YOLO model created from main.py - change text to your relative path
model = YOLO("best.pt")

# Uncomment this line to accelerate inference. Note that this may cause errors in some setups.
#model.half()
class_names = ['Basketball', 'Basketball Hoop']
device = get_device()

frame_count = 0
ball_pos = []  # array of tuples ((x_pos, y_pos), frame count, width, height, conf)
hoop_pos = []  # array of tuples ((x_pos, y_pos), frame count, width, height, conf)
curr_frame = None
makes = 0
attempts = 0
up = False
down = False
up_frame = 0
down_frame = 0
fade_frames = 20
fade_counter = 0
overlay_color = (0, 0, 0)
overlay_text = "Waiting..."

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def resize_and_pad(image, target_size, pad_color=(0, 0, 0)):
    """
    Resizes an image to fit within target_size while maintaining aspect ratio,
    and pads the remaining space with a specified color.

    Args:
        image (numpy.ndarray): The input image.
        target_size (tuple): A tuple (width, height) representing the desired output size.
        pad_color (tuple): A tuple (B, G, R) representing the padding color.

    Returns:
        numpy.ndarray: The resized and padded image.
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # Calculate scaling factor to maintain aspect ratio
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize the image
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Calculate padding amounts
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top

    # Add padding
    padded_image = cv2.copyMakeBorder(
        resized_image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=pad_color
    )

    return padded_image

def process_video(video_path):
    global frame_count, ball_pos, hoop_pos, curr_frame, makes, attempts, up, down, up_frame, down_frame, fade_frames, fade_counter, overlay_color, overlay_text

    # Load the YOLO model created from main.py - change text to your relative path
    overlay_text = "Waiting..."
    
    # Use video - replace text with your video path
    cap = cv2.VideoCapture(video_path)

    ball_pos = []  # array of tuples ((x_pos, y_pos), frame count, width, height, conf)
    hoop_pos = []  # array of tuples ((x_pos, y_pos), frame count, width, height, conf)	
    frame_count = 0

    makes = 0
    attempts = 0

    # Used to detect shots (upper and lower region)
    up = False
    down = False
    up_frame = 0
    down_frame = 0

    # Used for green and red colors after make/miss
    fade_frames = 20
    fade_counter = 0
    overlay_color = (0, 0, 0)


    while cap.isOpened():
        success, curr_frame = cap.read()
        if not success:
            break

        results = model(curr_frame, stream=True, device=device)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100

                # Class Name
                cls = int(box.cls[0])
                current_class = class_names[cls]

                center = (int(x1 + w / 2), int(y1 + h / 2))

                # Only create ball points if high confidence or near hoop
                if (conf > .3 or (in_hoop_region(center, hoop_pos) and conf > 0.15)) and current_class == "Basketball":
                    ball_pos.append((center, frame_count, w, h, conf))
                    cvzone.cornerRect(curr_frame, (x1, y1, w, h))

                # Create hoop points if high confidence
                if conf > .5 and current_class == "Basketball Hoop":
                    hoop_pos.append((center, frame_count, w, h, conf))
                    cvzone.cornerRect(curr_frame, (x1, y1, w, h))

        clean_motion()
        shot_detection()
        display_score()
        frame_count += 1

        target_dimensions = (640, 480)
        padding_color = (128, 128, 128)  # Grey color

        result_image = resize_and_pad(curr_frame, target_dimensions, padding_color)

        _, buffer = cv2.imencode('.jpg', result_image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


def clean_motion():
    global frame_count, ball_pos, hoop_pos, curr_frame, makes, attempts, up, down, up_frame, down_frame, fade_frames, fade_counter, overlay_color
    # Clean and display ball motion
    ball_pos = clean_ball_pos(ball_pos, frame_count)
    for i in range(0, len(ball_pos)):
        cv2.circle(curr_frame, ball_pos[i][0], 2, (0, 0, 255), 2)

    # Clean hoop motion and display current hoop center
    if len(hoop_pos) > 1:
        hoop_pos = clean_hoop_pos(hoop_pos)
        cv2.circle(curr_frame, hoop_pos[-1][0], 2, (128, 128, 0), 2)

def shot_detection():
    global frame_count, ball_pos, hoop_pos, curr_frame, makes, attempts, up, down, up_frame, down_frame, fade_frames, fade_counter, overlay_color, overlay_text

    if len(hoop_pos) > 0 and len(ball_pos) > 0:
        # Detecting when ball is in 'up' and 'down' area - ball can only be in 'down' area after it is in 'up'
        if not up:
            up = detect_up(ball_pos, hoop_pos)
            if up:
                up_frame = ball_pos[-1][1]

        if up and not down:
            down = detect_down(ball_pos, hoop_pos)
            if down:
                down_frame = ball_pos[-1][1]

        # If ball goes from 'up' area to 'down' area in that order, increase attempt and reset
        if frame_count % 10 == 0:
            if up and down and up_frame < down_frame:
                attempts += 1
                up = False
                down = False

                # If it is a make, put a green overlay and display "完美"
                if score(ball_pos, hoop_pos):
                    makes += 1
                    overlay_color = (0, 255, 0)  # Green for make
                    overlay_text = "Make"
                    fade_counter = fade_frames

                else:
                    overlay_color = (255, 0, 0)  # Red for miss
                    overlay_text = "Miss"
                    fade_counter = fade_frames

def display_score():
    global frame_count, ball_pos, hoop_pos, curr_frame, makes, attempts, up, down, up_frame, down_frame, fade_frames, fade_counter, overlay_color, overlay_text

    # Add text
    text = str(makes) + " / " + str(attempts)
    cv2.putText(curr_frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
    cv2.putText(curr_frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)

    # Add overlay text for shot result if it exists
    if True:
        # Calculate text size to position it at the right top corner
        (text_width, text_height), _ = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 3, 6)
        text_x = curr_frame.shape[1] - text_width - 40  # Right alignment with some margin
        text_y = 100  # Top margin

        # Display overlay text with color (overlay_color)
        cv2.putText(curr_frame, overlay_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 3,
                    overlay_color, 6)
        # cv2.putText(curr_frame, overlay_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)

    # Gradually fade out color after shot
    if fade_counter > 0:
        alpha = 0.2 * (fade_counter / fade_frames)
        curr_frame = cv2.addWeighted(curr_frame, 1 - alpha, np.full_like(curr_frame, overlay_color), alpha, 0)
        fade_counter -= 1

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'})

    file = request.files['video']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    return jsonify({'success': True, 'path': file_path})



@app.route('/video_feed')
def video_feed():
    video_path = request.args.get("path")
    if not video_path or not os.path.exists(video_path):
        return "Video not found", 404
    return Response(process_video(video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
