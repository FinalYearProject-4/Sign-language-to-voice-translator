# Full Integrated Code with Number Detection

from itertools import count
import time
import speech_recognition as sr
import numpy as np
import cv2
import os
from PIL import Image, ImageTk
import tkinter as tk
import string
import pickle
import mediapipe as mp
import Levenshtein
import pyttsx3

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

speak("Welcome to Sign Language Detection App")

def center_window(root, width, height):
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x_coordinate = (screen_width - width) // 2
    y_coordinate = (screen_height - height) // 2
    root.geometry(f"{width}x{height}+{x_coordinate}+{y_coordinate}")

def custom_buttonbox(msg, choices):
    root = tk.Tk()
    root.title("Sign Language Detection System")

    frame = tk.Frame(root)
    frame.pack(padx=10, pady=10)

    msg_label = tk.Label(frame, text=msg, font=("Helvetica", 16, "bold"))
    msg_label.pack(pady=20)

    buttonbox = tk.Frame(frame)
    buttonbox.pack(pady=10)

    for choice in choices:
        button = tk.Button(buttonbox, text=choice, font=("Helvetica", 14), command=lambda c=choice: on_button_click(root, c))
        button.pack(side=tk.LEFT, padx=10)

    try:
        img = Image.open("logo.png")
        Image.Resampling.LANCZOS

        logo = ImageTk.PhotoImage(img)
        logo_label = tk.Label(frame, image=logo)
        logo_label.image = logo
        logo_label.pack(pady=20)
    except Exception as e:
        print(f"Could not load logo image: {e}")

    center_window(root, 600, 400)
    root.mainloop()

def on_button_click(root, choice):
    root.destroy()
    if choice == "Voice To Sign":
        func()
    elif choice == "Sign Detection":
        signDetection()
    elif choice == "Number Detection":
        numberDetection()
    elif choice == "Exit":
        quit()

def find_closest_match(input_text, gesture_list):
    min_distance = float('inf')
    closest_match = None
    for gesture in gesture_list:
        distance = Levenshtein.distance(input_text, gesture)
        if distance < min_distance:
            min_distance = distance
            closest_match = gesture
    return closest_match if min_distance / max(len(input_text), len(closest_match)) < 0.4 else None

def func():
    r = sr.Recognizer()
    isl_gif = [...]  # Add your phrases here

    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        print("Listening...")
        audio = r.listen(source)

        try:
            a = r.recognize_google(audio).lower()
            print('You Said:', a)

            closest_match = find_closest_match(a, isl_gif)
            if closest_match:
                print(f"Closest match found: {closest_match}")
                a = closest_match.lower()

            for c in string.punctuation:
                a = a.replace(c, "")

            if a in ['goodbye', 'good bye', 'bye']:
                print("Goodbye!")
                return

            elif a in isl_gif:
                class ImageLabel(tk.Label):
                    def load(self, im):
                        if isinstance(im, str):
                            im = Image.open(im)
                        self.loc = 0
                        self.frames = []

                        try:
                            for _ in count(1):
                                self.frames.append(ImageTk.PhotoImage(im.copy()))
                                im.seek(_)
                        except EOFError:
                            pass

                        self.delay = im.info.get('duration', 100)

                        if len(self.frames) == 1:
                            self.config(image=self.frames[0])
                        else:
                            self.next_frame()

                    def unload(self):
                        self.config(image=None)
                        self.frames = None

                    def next_frame(self):
                        if self.frames:
                            self.loc = (self.loc + 1) % len(self.frames)
                            self.config(image=self.frames[self.loc])
                            self.after(self.delay, self.next_frame)

                root = tk.Tk()
                root.eval('tk::PlaceWindow . center')
                lbl = ImageLabel(root)
                lbl.pack()
                lbl.load(f'ISL_Gifs/{a}.gif')
                root.mainloop()

            else:
                print("Gesture not found.")

        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print(f"Request error: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

def signDetection():
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']

    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=1)
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    labels_dict = {i: chr(65 + i) for i in range(26)}
    
    prev_prediction = None
    prediction_count = 0
    stable_threshold = 10
    last_spoken = ""

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                data_aux = []
                x_, y_ = [], []

                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                for lm in hand_landmarks.landmark:
                    data_aux.extend([lm.x - min(x_), lm.y - min(y_)])

                x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
                x2, y2 = int(max(x_) * W) + 10, int(max(y_) * H) + 10

                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                if predicted_character == prev_prediction:
                    prediction_count += 1
                else:
                    prediction_count = 0
                prev_prediction = predicted_character

                if prediction_count > stable_threshold:
                    cv2.putText(frame, f"Detected: {predicted_character}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
                    
                    if predicted_character != last_spoken:
                        speak(f"The detected letter is {predicted_character}")
                        last_spoken = predicted_character

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 3)

        cv2.imshow('frame', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def calculate_distance(p1, p2):
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

def get_number(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_IP]
    index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]
    middle_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_mcp = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP]

    fingers_up = [
        thumb_tip.x < thumb_ip.x,
        index_tip.y < index_mcp.y,
        middle_tip.y < middle_mcp.y,
        ring_tip.y < middle_mcp.y,
        pinky_tip.y < middle_mcp.y
    ]

    if fingers_up == [False, True, False, False, False]:
        return "1"
    elif fingers_up == [False, True, True, False, False]:
        return "2"
    elif fingers_up == [True, True, True, False, False]:
        return "3"
    elif fingers_up == [False, True, True, True, True]:
        return "4"
    elif fingers_up == [True, True, True, True, True]:
        return "5"
    elif calculate_distance(thumb_tip, pinky_tip) < 0.05 and all(fingers_up[1:4]):
        return "6"
    elif calculate_distance(thumb_tip, ring_tip) < 0.05 and all(fingers_up[1:3] + [fingers_up[4]]):
        return "7"
    elif calculate_distance(thumb_tip, middle_tip) < 0.05 and fingers_up[1] and fingers_up[3] and fingers_up[4]:
        return "8"
    elif calculate_distance(thumb_tip, index_tip) < 0.05 and all(fingers_up[2:]):
        return "9"
    elif fingers_up == [True, False, False, False, False]:
        return "10"
    else:
        return "No Recognized Number"

def numberDetection():
    cap = cv2.VideoCapture(0)
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.9,
        min_tracking_confidence=0.9
    )
    mp_drawing = mp.solutions.drawing_utils

    last_number = ""
    confirmed_number = ""
    same_count = 0
    stable_threshold = 5  # How many consistent frames needed to confirm

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        number = "No Recognized Number"

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            number = get_number(hand_landmarks)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

            if number == last_number and number != "No Recognized Number":
                same_count += 1
            else:
                same_count = 0
            last_number = number

            if same_count >= stable_threshold:
                if confirmed_number != number:
                    confirmed_number = number
                    speak(f"The number is {number}")
                same_count = 0  # Reset to avoid repetition

        # Always show the last confirmed number
        if confirmed_number != "":
            cv2.putText(frame, f"Number: {confirmed_number}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Number Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ➤ Launch the App
while True:
    message = "SIGN LANGUAGE DETECTION SYSTEM"
    choices = ["Sign Detection", "Number Detection", "Exit"]
    custom_buttonbox(message,choices)
