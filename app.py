from flask import Flask, render_template, Response, session
import cv2
import dlib
import math
import numpy as np
import tensorflow as tf
from datetime import datetime
from flask import jsonify


from flask import Flask, request, flash, redirect, url_for, render_template

from collections import deque
from constants1 import TOTAL_FRAMES, VALID_WORD_THRESHOLD, NOT_TALKING_THRESHOLD, PAST_BUFFER_SIZE, LIP_WIDTH, LIP_HEIGHT
import os
import cv2
import dlib
import math
import json
import statistics
from PIL import Image
import imageio.v2 as imageio
import numpy as np
import csv
from collections import deque
import tensorflow as tf
import sys
sys.path.append(r'C:\Users\HP\Desktop\cnn\Computer-Vision-Lip-Reading-2.0-main\data_collection')
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'default_secret_key')

CONTACT_FILE = 'contact.csv'
LOGIN_ACTIVITY_FILE = 'login_activity.csv'


# Write contact submission to CSV
# Write contact submission to CSV
def write_contact_submission(contact_data):
    file_exists = os.path.isfile(CONTACT_FILE)
    with open(CONTACT_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Name', 'Email', 'Message'])  # Write header if file does not exist
        writer.writerow(contact_data)



# Read contact submissions from CSV
def read_contact_submissions():
    contacts = []
    if os.path.exists(CONTACT_FILE):
        with open(CONTACT_FILE, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                contacts.append(row)
    return contacts

# File path for storing user data
USER_FILE = 'user1.csv'


def read_users():
    users = {}
    if os.path.exists(USER_FILE):
        with open(USER_FILE, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                users[row[1]] = row  # Store the entire row indexed by username (email)
    return users




# Write new user to CSV
def write_user(user_data):
    with open(USER_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(user_data)

def write_login_activity(email):
    with open(LOGIN_ACTIVITY_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([email, datetime.now().strftime('%Y-%m-%d')])

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/videos')
def videos():
    return render_template('videos.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        
        # Save contact form data to CSV
        write_contact_submission([name, email, message])
        
        flash('Thank you for contacting us! We will get back to you shortly.', 'success')
        return redirect(url_for('contact'))
    
    return render_template('contact.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        location = request.form['location']
        gender = request.form['gender']
        deaf = request.form['deaf']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('register'))

        users = read_users()
        if email in users:
            flash('Email already exists', 'danger')
        else:
            user_data = [name, email, phone, location, gender, deaf, password]
            write_user(user_data)
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('userlogin'))
    return render_template('register.html')

@app.route('/userlogin', methods=['GET', 'POST'])
def userlogin():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        users = read_users()
        if email in users and users[email][-1] == password:
            session['username'] = users[email][0]
            write_login_activity(email)  # Record login activity
            flash('Login successful!', 'success')
            return redirect(url_for('main'))
        else:
            flash('Invalid credentials, please try again.', 'danger')
    return render_template('userlogin.html')

def get_active_days(email):
    active_days = set()
    if os.path.exists(LOGIN_ACTIVITY_FILE):
        with open(LOGIN_ACTIVITY_FILE, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0] == email:
                    active_days.add(row[1])
    return len(active_days)


@app.route('/index')
def main():
    username = session.get('username')
    email = None
    users = read_users()
    for user_email, user_data in users.items():
        if user_data[0] == username:
            email = user_email
            break
    active_days = get_active_days(email) if email else 0
    return render_template('index.html', username=username, active_days=active_days)

detector2 = dlib.get_frontal_face_detector()

# Load the predictor
model_weights_path2 = "C:/Users/HP/Desktop/cnn/Computer-Vision-Lip-Reading-2.0-main/model/face_weights.dat"
predictor2 = dlib.shape_predictor(model_weights_path2)


    

@app.route('/adminpage')
def adminpage():
    users = []
    deaf_users = []
    non_deaf_users = []

    # Read all users from the CSV file
    if os.path.exists(USER_FILE):
        with open(USER_FILE, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                users.append(row)
                if row['deaf'].lower() == 'yes':
                    deaf_users.append(row)
                else:
                    non_deaf_users.append(row)

    contacts = read_contact_submissions()
    return render_template('adminpage.html', contacts=contacts, users=users, deaf_users=deaf_users, non_deaf_users=non_deaf_users)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/adminlogin')
def adminlogin():
    return render_template('adminlogin.html')

@app.route('/predict')
def prediction():
    from constants1 import TOTAL_FRAMES, VALID_WORD_THRESHOLD, NOT_TALKING_THRESHOLD, PAST_BUFFER_SIZE, LIP_WIDTH, LIP_HEIGHT
    label_dict = {6: 'hello', 5: 'dog', 10: 'my', 12: 'you', 9: 'lips', 3: 'cat', 11: 'read', 0: 'a', 4: 'demo', 7: 'here', 8: 'is', 1: 'bye', 2: 'can'}
    count = 0
    input_shape = (TOTAL_FRAMES, 80, 112, 3)
    model = tf.keras.Sequential([
        tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling3D((2, 2, 2)),
        tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu'),
        tf.keras.layers.MaxPooling3D((2, 2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(label_dict), activation='softmax')
    ])
    model_weights_path1 = "C:/Users/HP/Desktop/cnn/Computer-Vision-Lip-Reading-2.0-main/model/model_weights.h5"
    model.load_weights(model_weights_path1, by_name=True)
    detector = dlib.get_frontal_face_detector()
    model_weights_path2 = "C:/Users/HP/Desktop/cnn/Computer-Vision-Lip-Reading-2.0-main/model/face_weights.dat"
    predictor = dlib.shape_predictor(model_weights_path2)
    cap = cv2.VideoCapture(0)
    curr_word_frames = []
    not_talking_counter = 0
    first_word = True
    labels = []
    past_word_frames = deque(maxlen=PAST_BUFFER_SIZE)
    ending_buffer_size = 5
    predicted_word_label = None
    draw_prediction = False
    spoken_already = []
    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right() 
            y2 = face.bottom()
            landmarks = predictor(image=gray, box=face)
            mouth_top = (landmarks.part(51).x, landmarks.part(51).y)
            mouth_bottom = (landmarks.part(57).x, landmarks.part(57).y)
            lip_distance = math.hypot(mouth_bottom[0] - mouth_top[0], mouth_bottom[1] - mouth_top[1])
            lip_left = landmarks.part(48).x
            lip_right = landmarks.part(54).x
            lip_top = landmarks.part(50).y
            lip_bottom = landmarks.part(58).y

            width_diff = LIP_WIDTH - (lip_right - lip_left)
            height_diff = LIP_HEIGHT - (lip_bottom - lip_top)
            pad_left = width_diff // 2
            pad_right = width_diff - pad_left
            pad_top = height_diff // 2
            pad_bottom = height_diff - pad_top

            pad_left = min(pad_left, lip_left)
            pad_right = min(pad_right, frame.shape[1] - lip_right)
            pad_top = min(pad_top, lip_top)
            pad_bottom = min(pad_bottom, frame.shape[0] - lip_bottom)

            lip_frame = frame[lip_top - pad_top:lip_bottom + pad_bottom, lip_left - pad_left:lip_right + pad_right]
            lip_frame = cv2.resize(lip_frame, (LIP_WIDTH, LIP_HEIGHT))

            lip_frame_lab = cv2.cvtColor(lip_frame, cv2.COLOR_BGR2LAB)
         #Apply contrast stretching to the L channel of the LAB image
            l_channel, a_channel, b_channel = cv2.split(lip_frame_lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3,3))
            l_channel_eq = clahe.apply(l_channel)

            # Merge the equalized L channel with the original A and B channels
            lip_frame_eq = cv2.merge((l_channel_eq, a_channel, b_channel))
            lip_frame_eq = cv2.cvtColor(lip_frame_eq, cv2.COLOR_LAB2BGR)
            lip_frame_eq= cv2.GaussianBlur(lip_frame_eq, (7, 7), 0)
            lip_frame_eq = cv2.bilateralFilter(lip_frame_eq, 5, 75, 75)
            kernel = np.array([[-1,-1,-1],
                        [-1, 9,-1],
                        [-1,-1,-1]])
            lip_frame_eq = cv2.filter2D(lip_frame_eq, -1, kernel)
            lip_frame_eq= cv2.GaussianBlur(lip_frame_eq, (5, 5), 0)
            lip_frame = lip_frame_eq

            for n in range(48, 61):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(img=frame, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)

            if lip_distance > 45:
                cv2.putText(frame, "Talking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                curr_word_frames += [lip_frame.tolist()]
                not_talking_counter = 0
                draw_prediction = False
            else:
                cv2.putText(frame, "Not talking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                not_talking_counter += 1
                if not_talking_counter >= NOT_TALKING_THRESHOLD and len(curr_word_frames) + PAST_BUFFER_SIZE == TOTAL_FRAMES:
                    curr_word_frames = list(past_word_frames) + curr_word_frames
                    curr_data = np.array([curr_word_frames[:input_shape[0]]])
                    print("*********", curr_data.shape)
                    print(spoken_already)
                    prediction = model.predict(curr_data)
                    prob_per_class = []
                    for i in range(len(prediction[0])):
                        prob_per_class.append((prediction[0][i], label_dict[i]))
                    sorted_probs = sorted(prob_per_class, key=lambda x: x[0], reverse=True)
                    for prob, label in sorted_probs:
                        print(f"{label}: {prob:.3f}")
                    predicted_class_index = np.argmax(prediction)
                    while label_dict[predicted_class_index] in spoken_already:
                        prediction[0][predicted_class_index] = 0
                        predicted_class_index = np.argmax(prediction)
                    predicted_word_label = label_dict[predicted_class_index]
                    spoken_already.append(predicted_word_label)

                    print("FINISHED!", predicted_word_label)
                    draw_prediction = True
                    count = 0
                    curr_word_frames = []
                    not_talking_counter = 0
                elif not_talking_counter < NOT_TALKING_THRESHOLD and len(curr_word_frames) + PAST_BUFFER_SIZE < TOTAL_FRAMES and len(curr_word_frames) > VALID_WORD_THRESHOLD:
                    curr_word_frames += [lip_frame.tolist()]
                    not_talking_counter = 0
                elif len(curr_word_frames) < VALID_WORD_THRESHOLD or (not_talking_counter >= NOT_TALKING_THRESHOLD and len(curr_word_frames) + PAST_BUFFER_SIZE > TOTAL_FRAMES):
                    curr_word_frames = []
                past_word_frames+= [lip_frame.tolist()]
                if len(past_word_frames) > PAST_BUFFER_SIZE:
                    past_word_frames.pop(0)
        if(draw_prediction and count < 20):
            count += 1
            cv2.putText(frame, predicted_word_label, (50 ,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
        cv2.imshow(winname="Mouth", mat=frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            spoken_already = []
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    return render_template('predict.html')

# Constants
TOTAL_FRAMES = 30  # Total number of frames per word
VALID_WORD_THRESHOLD = 5  # Minimum number of frames to consider a word valid
NOT_TALKING_THRESHOLD = 10  # Threshold for considering someone not talking
PAST_BUFFER_SIZE = 5  # Size of the buffer for storing previous frames
LIP_WIDTH = 80  # Width of the lip region
LIP_HEIGHT = 112  # Height of the lip region

ORANGE = (0, 180, 255)


# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
model_weights_path1 = "C:/Users/HP/Desktop/cnn/Computer-Vision-Lip-Reading-2.0-main/model/face_weights.dat"
predictor = dlib.shape_predictor(model_weights_path1)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Function to save collected words
def save_all_words(all_words):
    print("Saving words into directories...")
    output_dir = "C:/Users/HP/Desktop/cnn/Computer-Vision-Lip-Reading-2.0-main/collected_data"
    next_dir_number = 1

    for i, word_frames in enumerate(all_words):
        label = "word_" + str(i)
        word_folder = os.path.join(output_dir, label + "_" + f"{next_dir_number}")

        while os.path.exists(word_folder):
            next_dir_number += 1
            word_folder = os.path.join(output_dir, label + "_" + f"{next_dir_number}")
        
        os.makedirs(word_folder)

        # Save frames as images
        for j, img_data in enumerate(word_frames):
            img = Image.fromarray(np.uint8(img_data))
            img_path = os.path.join(word_folder, f"{j}.png")
            img.save(img_path)

        # Save frames as video
        video_path = os.path.join(word_folder, "video.mp4")
        imageio.mimsave(video_path, word_frames, fps=30)
        next_dir_number += 1

# Collect data function
def collect_data():
    all_words = []
    not_talking_counter = 0
    determining_lip_distance = 50
    lip_distances = []
    LIP_DISTANCE_THRESHOLD = None
    curr_word_frames = []

    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(image=gray, box=face)
            mouth_top = (landmarks.part(51).x, landmarks.part(51).y)
            mouth_bottom = (landmarks.part(57).x, landmarks.part(57).y)
            lip_distance = math.hypot(mouth_bottom[0] - mouth_top[0], mouth_bottom[1] - mouth_top[1])

            lip_left = landmarks.part(48).x
            lip_right = landmarks.part(54).x
            lip_top = landmarks.part(50).y
            lip_bottom = landmarks.part(58).y

            if determining_lip_distance != 0 and LIP_DISTANCE_THRESHOLD is None:
                width_diff = LIP_WIDTH - (lip_right - lip_left)
                height_diff = LIP_HEIGHT - (lip_bottom - lip_top)
                pad_left = width_diff // 2
                pad_right = width_diff - pad_left
                pad_top = height_diff // 2
                pad_bottom = height_diff - pad_top

                pad_left = min(pad_left, lip_left)
                pad_right = min(pad_right, frame.shape[1] - lip_right)
                pad_top = min(pad_top, lip_top)
                pad_bottom = min(pad_bottom, frame.shape[0] - lip_bottom)

                lip_frame = frame[lip_top - pad_top : lip_bottom + pad_bottom, lip_left - pad_left : lip_right + pad_right]
                lip_frame = cv2.resize(lip_frame, (LIP_WIDTH, LIP_HEIGHT))

                lip_frame_lab = cv2.cvtColor(lip_frame, cv2.COLOR_BGR2LAB)
                l_channel, a_channel, b_channel = cv2.split(lip_frame_lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3,3))
                l_channel_eq = clahe.apply(l_channel)
                lip_frame_eq = cv2.merge((l_channel_eq, a_channel, b_channel))
                lip_frame_eq = cv2.cvtColor(lip_frame_eq, cv2.COLOR_LAB2BGR)
                lip_frame_eq = cv2.GaussianBlur(lip_frame_eq, (7, 7), 0)
                lip_frame_eq

                lip_distances.append(lip_distance)
                if determining_lip_distance == 0:
                    LIP_DISTANCE_THRESHOLD = sum(lip_distances) / len(lip_distances)
                    determining_lip_distance = -1
                else:
                    determining_lip_distance -= 1

            else:
                if LIP_DISTANCE_THRESHOLD is not None:
                    if lip_distance > LIP_DISTANCE_THRESHOLD:  # Person is talking
                        cv2.putText(frame, "Talking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        curr_word_frames.append(frame)  # Example: Append the current frame to curr_word_frames
                        not_talking_counter = 0
                        cv2.putText(frame, "RECORDING WORD RIGHT NOW", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, ORANGE, 2)
                    else:
                        cv2.putText(frame, "Not talking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        not_talking_counter += 1
                        if not_talking_counter >= NOT_TALKING_THRESHOLD and len(curr_word_frames) == TOTAL_FRAMES:
                            all_words.append(curr_word_frames)  # Example: Append curr_word_frames to all_words
                            curr_word_frames = []
                            not_talking_counter = 0
                        elif not_talking_counter < NOT_TALKING_THRESHOLD and len(curr_word_frames) < TOTAL_FRAMES:
                            pass
                        else:
                            pass
                else:
                    pass

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    save_all_words(all_words)
    cap.release()
    cv2.destroyAllWindows()

@app.route('/collects', methods=['GET', 'POST'])
def collects():
    if request.method == 'POST':
        collect_data()
        return jsonify({'message': 'Data collected successfully!'})
    else:
        return render_template('collects.html')






if __name__ == '__main__':
    app.run(debug=True)
