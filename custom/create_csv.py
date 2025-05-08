import os
import csv
import argparse
from pathlib import Path


def create_video_feature_csv(root_dir, output_csv, feature_dir):
    """
    Create a CSV file mapping video paths to feature paths.

    Args:
        root_dir (str): Directory containing folders with videos
        output_csv (str): Path where the CSV file will be saved
        feature_dir (str): Directory where features are/will be stored
    """
    # Supported video extensions
    video_extensions = ['.mp4', '.webm', '.mkv', '.avi', '.mov', '.flv', '.wmv']

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # List to store video paths
    video_paths = []

    # First, collect all video paths
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            file_ext = os.path.splitext(filename)[1].lower()

            # Check if the file is a video
            if file_ext in video_extensions:
                video_path = os.path.join(dirpath, filename)
                video_paths.append(video_path)

    # Sort the paths to ensure consistent numbering
    video_paths.sort()

    # Create video_feature_pairs with numbered feature paths
    video_feature_pairs = []
    for i, video_path in enumerate(video_paths, 1):
        # Create numbered feature path (video1.npz, video2.npz, etc.)
        feature_path = os.path.join(feature_dir, f"video{i}.npz")
        video_feature_pairs.append((video_path, feature_path))

    # Write to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write header
        csv_writer.writerow(['video_path', 'feature_path'])

        # Write data
        for video_path, feature_path in video_feature_pairs:
            csv_writer.writerow([video_path, feature_path])

    print(f"CSV created successfully at {output_csv}")
    print(f"Found {len(video_feature_pairs)} videos")


def main():
    parser = argparse.ArgumentParser(description='Create a CSV mapping videos to feature paths')
    parser.add_argument('--root_dir', type=str, default=r"D:\dataset_abyss\filtered_videos_1433",
                        help='Directory containing folders with videos')
    parser.add_argument('--output_csv', type=str, default='csv/video_features.csv',
                        help='Path where the CSV file will be saved')
    parser.add_argument('--feature_dir', type=str, default='/output/slowfast_features',
                        help='Directory where features are/will be stored')

    args = parser.parse_args()

    create_video_feature_csv(args.root_dir, args.output_csv, args.feature_dir)


if __name__ == "__main__":
    main()