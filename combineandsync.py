import os
import sys
import pandas as pd
import numpy as np
import cv2

def load_csv_files(folder_path):
    files = os.listdir(folder_path)
    gps_files = [f for f in files if "GPS9_sample_final.csv" in f]
    accl_files = [f for f in files if "ACCL_sample_final.csv" in f]
    gyro_files = [f for f in files if "GYRO_sample_final.csv" in f]
    
    if not gps_files or not accl_files or not gyro_files:
        raise FileNotFoundError("One or more required CSV files are missing")
    
    gps_df = pd.read_csv(os.path.join(folder_path, gps_files[0]))
    accl_df = pd.read_csv(os.path.join(folder_path, accl_files[0]))
    gyro_df = pd.read_csv(os.path.join(folder_path, gyro_files[0]))

    return gps_df, accl_df, gyro_df

def combine_data(gps_df, accl_df, gyro_df):
    combined_df = pd.merge_asof(gps_df.sort_values('Sample time [seg]'),
                                accl_df.sort_values('Sample time [seg]'),
                                on='Sample time [seg]',
                                direction='nearest')

    combined_df = pd.merge_asof(combined_df.sort_values('Sample time [seg]'),
                                gyro_df.sort_values('Sample time [seg]'),
                                on='Sample time [seg]',
                                direction='nearest')

    return combined_df

def extract_frames(video_path, frames_output_path):
    if not os.path.exists(frames_output_path):
        os.makedirs(frames_output_path)

    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(frame_rate // 6)

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame_id = int(count // frame_interval)
            frame_filename = f"frame_{frame_id:06d}.jpeg"
            frame_filepath = os.path.join(frames_output_path, frame_filename)
            cv2.imwrite(frame_filepath, frame)
        count += 1

    cap.release()

def sync_frames_to_data(combined_df, frames_output_path):
    frame_files = sorted([f for f in os.listdir(frames_output_path) if f.endswith(('.jpeg', '.JPEG'))])
    frame_timestamps = np.linspace(0, combined_df['Sample time [seg]'].max(), len(frame_files))

    frames_df = pd.DataFrame({
        'Frame': [os.path.join(frames_output_path, f) for f in frame_files],
        'Image Timestamp [s]': frame_timestamps
    })

    synced_df = pd.merge_asof(combined_df.sort_values('Sample time [seg]'),
                              frames_df,
                              left_on='Sample time [seg]',
                              right_on='Image Timestamp [s]',
                              direction='nearest')

    return synced_df

def main(folder_path, video_path):
    gps_df, accl_df, gyro_df = load_csv_files(folder_path)
    combined_df = combine_data(gps_df, accl_df, gyro_df)

    frames_output_path = os.path.join(folder_path, 'frames')
    extract_frames(video_path, frames_output_path)

    synced_df = sync_frames_to_data(combined_df, frames_output_path)
    synced_df.to_csv(os.path.join(folder_path, 'combined_metrics.csv'), index=False)
    print(f"Combined and synced data saved to {os.path.join(folder_path, 'combined_metrics.csv')}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python combine_and_sync.py <folder_path> <video_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    video_path = sys.argv[2]

    main(folder_path, video_path)
