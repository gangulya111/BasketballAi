from flask import Flask, render_template, Response, request, jsonify
import cv2, os, numpy as np
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import time
import cvzone
from collections import deque

# Load YOLO models
ball_model = YOLO("basketballModel.pt")
pose_model = YOLO("yolov8s-pose.pt")


app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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


body_index = {
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16
}

def calculate_angle(a, b, c):
    """Calculate angle between 3 points (a-b-c) in degrees."""
    a, b, c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

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

	# Stats tracking
	ball_heights = []
	dribble_times = deque(maxlen=20)


	frame_buffer = deque(maxlen=30)
	save_frames = 60
	frame_save_counter = 0
	saving = False
	out = None

	# Initialize counters and trackers
	dribble_count = 0
	step_count = 0
	prev_x_center = None
	prev_y_center = None
	prev_left_ankle_y = None
	prev_right_ankle_y = None
	prev_delta_y = None
	ball_not_detected_frames = 0
	max_ball_not_detected_frames = 20
	dribble_threshold = 18
	step_threshold = 5
	min_wait_frames = 7
	wait_frames = 0
	travel_detected = False
	travel_timestamp = None
	total_dribble_count = 0
	total_step_count = 0
	cap = cv2.VideoCapture(video_path)
	
	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))
	fps = cap.get(cv2.CAP_PROP_FPS)
	fourcc = cv2.VideoWriter_fourcc(*"mp4v")

	# Create directories
	os.makedirs("travel_footage", exist_ok=True)
	os.makedirs("full_output", exist_ok=True)

	# Output writer
	full_output_path = os.path.join("full_output", f"full_motion_{time.strftime('%Y%m%d-%H%M%S')}.mp4")
	full_output = cv2.VideoWriter(full_output_path, fourcc, fps, (frame_width, frame_height))

	while cap.isOpened():
		success, frame = cap.read()
		if not success:
		    break

		frame_buffer.append(frame)

		# Ball detection
		ball_results_list = ball_model(frame, verbose=False, conf=0.5)
		ball_detected = False

		for results in ball_results_list:
		    for bbox in results.boxes.xyxy:
		        x1, y1, x2, y2 = bbox[:4]
		        x_center = (x1 + x2) / 2
		        y_center = (y1 + y2) / 2
		        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
		        w, h = x2 - x1, y2 - y1

		        ball_heights.append(y_center)
		        if len(ball_heights) > 100:
		            ball_heights.pop(0)

		        if prev_y_center is not None:
		            delta_y = y_center - prev_y_center
		            if (
		                prev_delta_y is not None
		                and prev_delta_y > dribble_threshold
		                and delta_y < -dribble_threshold
		            ):
		                dribble_count += 1
		                total_dribble_count += 1
		                dribble_times.append(time.time())

		            prev_delta_y = delta_y

		        prev_x_center = x_center
		        prev_y_center = y_center
		        ball_detected = True
		        ball_not_detected_frames = 0
		        cvzone.cornerRect(frame, (x1, y1, w, h))

		    #annotated_frame = results.plot()

		if not ball_detected:
		    ball_not_detected_frames += 1
		if ball_not_detected_frames >= max_ball_not_detected_frames:
		    step_count = 0

		# Pose detection
		pose_results = pose_model(frame, verbose=False, conf=0.5)
		rounded_results = np.round(pose_results[0].keypoints.xy.cpu().numpy(), 1)

		left_knee_angle = right_knee_angle = 0

		try:
		    left_hip = rounded_results[0][body_index["left_hip"]]
		    right_hip = rounded_results[0][body_index["right_hip"]]
		    left_knee = rounded_results[0][body_index["left_knee"]]
		    right_knee = rounded_results[0][body_index["right_knee"]]
		    left_ankle = rounded_results[0][body_index["left_ankle"]]
		    right_ankle = rounded_results[0][body_index["right_ankle"]]

		    if all(joint[2] > 0.5 for joint in [left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle]):
		        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
		        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

		        if prev_left_ankle_y is not None and prev_right_ankle_y is not None and wait_frames == 0:
		            left_diff = abs(left_ankle[1] - prev_left_ankle_y)
		            right_diff = abs(right_ankle[1] - prev_right_ankle_y)

		            if max(left_diff, right_diff) > step_threshold:
		                step_count += 1
		                total_step_count += 1
		                print(f"Step taken: {step_count}")
		                wait_frames = min_wait_frames

		        prev_left_ankle_y = left_ankle[1]
		        prev_right_ankle_y = right_ankle[1]

		        if wait_frames > 0:
		            wait_frames -= 1

		except Exception:
		    pass

		pose_annotated_frame = pose_results[0].plot(labels=False, conf=False)
		combined_frame = pose_annotated_frame #cv2.addWeighted(frame, 0.6, pose_annotated_frame, 0.4, 0)

		# Travel detection
		if ball_detected and step_count >= 2 and dribble_count == 0:
		    print("Travel detected!")
		    step_count = 0
		    travel_detected = True
		    travel_timestamp = time.time()
		    if not saving:
		        filename = os.path.join("travel_footage", f"travel_{time.strftime('%Y%m%d-%H%M%S')}.mp4")
		        out = cv2.VideoWriter(filename, fourcc, 9, (frame_width, frame_height))
		        for f in frame_buffer:
		            out.write(f)
		            frame_save_counter += 1
		        saving = True

		if travel_detected and time.time() - travel_timestamp > 3:
		    travel_detected = False
		    total_dribble_count = 0
		    total_step_count = 0

		if travel_detected:
		    blue_tint = np.full_like(combined_frame, (255, 0, 0), dtype=np.uint8)
		    combined_frame = cv2.addWeighted(combined_frame, 0.7, blue_tint, 0.3, 0)
		    cv2.putText(combined_frame, "Travel Detected!", (frame_width - 600, 150),
		                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4, cv2.LINE_AA)

		if dribble_count > 0:
		    step_count = 0
		    dribble_count = 0

		if saving:
		    out.write(combined_frame)
		    frame_save_counter += 1
		    if frame_save_counter >= save_frames:
		        saving = False
		        frame_save_counter = 0
		        out.release()

		# --- Simple Stats Section ---
		avg_ball_height = np.mean(ball_heights) if ball_heights else 0

		if len(dribble_times) >= 2:
		    total_time = dribble_times[-1] - dribble_times[0]
		    dribble_rate = (len(dribble_times) - 1) / total_time if total_time > 0 else 0
		else:
		    dribble_rate = 0

		# Draw stats box (top-left corner)
		#cv2.rectangle(combined_frame, (20, 20), (600, 200), (255, 255, 255), -1)
		cv2.putText(combined_frame, "STATS", (40, 60),
		            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3, cv2.LINE_AA)
		#cv2.putText(combined_frame, f"Avg Ball Height: {(avg_ball_height/10):.1f} cm", (40, 100),
		#            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3, cv2.LINE_AA)
		cv2.putText(combined_frame, f"Dribble Rate: {dribble_rate:.2f} / sec", (40, 140),
		            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3, cv2.LINE_AA)
		cv2.putText(combined_frame, f"Dribble Count: {total_dribble_count}", (40, 180),
		            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3, cv2.LINE_AA)
					
		target_dimensions = (640, 480)
		padding_color = (128, 128, 128)  # Grey color

		result_image = resize_and_pad(combined_frame, target_dimensions, padding_color)
			
		_, buffer = cv2.imencode('.jpg', result_image)
		frame = buffer.tobytes()
		yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
				
	cap.release()

@app.route('/video_feed')
def video_feed():
    video_path = request.args.get("path")
    if not video_path or not os.path.exists(video_path):
        return "Video not found", 404
    return Response(process_video(video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

