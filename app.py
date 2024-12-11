from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import threading

app = Flask(__name__)

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

sensitivity = 0.18
drowsy_count = 0
max_drowsy_count = 10
video_capture = cv2.VideoCapture(0)

# Calculate Eye Aspect Ratio (EAR)
def calculate_ear(landmarks, eye_indices):
    def distance(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    vertical1 = distance(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
    vertical2 = distance(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
    horizontal = distance(landmarks[eye_indices[0]], landmarks[eye_indices[3]])

    return (vertical1 + vertical2) / (2 * horizontal)

def process_frame():
    global drowsy_count

    ret, frame = video_capture.read()
    if not ret:
        return None

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    is_drowsy = False
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0]))
                         for lm in face_landmarks.landmark]

            left_eye = calculate_ear(landmarks, [33, 160, 158, 133, 153, 144])
            right_eye = calculate_ear(landmarks, [362, 385, 387, 263, 373, 380])
            ear = (left_eye + right_eye) / 2.0

            if ear < sensitivity:
                drowsy_count += 1
                if drowsy_count > max_drowsy_count:
                    is_drowsy = True
            else:
                drowsy_count = max(0, drowsy_count - 1)

    if is_drowsy:
        cv2.putText(frame, "DROWSY!!!", (250, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 11)


    _, jpeg = cv2.imencode('.jpg', frame)
    return jpeg.tobytes()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = process_frame()
            is_drowsy_state = drowsy_count > max_drowsy_count
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            yield (b'--state\r\n'
                   b'Content-Type: application/json\r\n\r\n'
                   + bytes(f'{{"is_drowsy": {is_drowsy_state}}}', 'utf-8') + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update_sensitivity', methods=['POST'])
def update_sensitivity():
    global sensitivity
    data = request.json
    sensitivity = float(data['sensitivity'])
    return jsonify({'status': 'success', 'sensitivity': sensitivity})

if __name__ == "__main__":
    app.run(debug=True, threaded=True)