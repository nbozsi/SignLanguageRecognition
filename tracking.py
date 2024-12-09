import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

# Initialize Mediapipe holistic model
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Function to extract landmarks from a frame
def extract_landmarks(results):
    frame_landmarks = []

    # Extract face landmarks (468 landmarks)
    if results.face_landmarks:
        face_landmarks = []
        for landmark in results.face_landmarks.landmark:
            face_landmarks.extend([landmark.x, landmark.y, landmark.z])
        frame_landmarks.extend(face_landmarks)
    else:
        frame_landmarks.extend([0] * 468 * 3)

    # Extract left hand landmarks (21 landmarks)
    if results.left_hand_landmarks:
        left_hand_landmarks = []
        for landmark in results.left_hand_landmarks.landmark:
            left_hand_landmarks.extend([landmark.x, landmark.y, landmark.z])
        frame_landmarks.extend(left_hand_landmarks)
    else:
        frame_landmarks.extend([0] * 21 * 3)

    # Extract right hand landmarks (21 landmarks)
    if results.right_hand_landmarks:
        right_hand_landmarks = []
        for landmark in results.right_hand_landmarks.landmark:
            right_hand_landmarks.extend([landmark.x, landmark.y, landmark.z])
        frame_landmarks.extend(right_hand_landmarks)
    else:
        frame_landmarks.extend([0] * 21 * 3)

    # Extract pose landmarks (33 landmarks)
    if results.pose_landmarks:
        pose_landmarks = []
        for landmark in results.pose_landmarks.landmark:
            pose_landmarks.extend([landmark.x, landmark.y, landmark.z])
        frame_landmarks.extend(pose_landmarks)
    else:
        frame_landmarks.extend([0] * 33 * 3)

    return frame_landmarks

# Process a single video and extract landmarks
def process_video(video_path, output_data_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    landmarks_all_frames = []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False) as holistic:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Convert the BGR image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Process the image and detect the landmarks
            results = holistic.process(image)

            # Extract landmarks
            frame_landmarks = extract_landmarks(results)
            landmarks_all_frames.append(frame_landmarks)

        # Release resources
        cap.release()

        # Convert to DataFrame
        df = pd.DataFrame(landmarks_all_frames)

        # Handle missing data by interpolation
        df.replace(0, np.nan, inplace=True)
        df.interpolate(method='linear', limit_direction='both', inplace=True)
        df.fillna(0, inplace=True)

        # Save landmarks to CSV
        df.to_csv(output_data_path, index=False)

# Main function to process all videos
def main():
    input_folder = './cropped'  # Replace with your input folder path
    output_data_folder = './landmark_data'  # Replace with your output data folder path

    os.makedirs(output_data_folder, exist_ok=True)

    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith('.mp4') or filename.endswith('.avi'):
                video_path = os.path.join(root, filename)

                # Construct relative path
                rel_path = os.path.relpath(video_path, input_folder)

                # Construct output path
                output_data_path = os.path.join(
                    output_data_folder, os.path.splitext(rel_path)[0] + '.csv')

                # Ensure output directory exists
                os.makedirs(os.path.dirname(output_data_path), exist_ok=True)

                print(f'Processing {video_path}')
                process_video(video_path, output_data_path)

if __name__ == '__main__':
    main()
