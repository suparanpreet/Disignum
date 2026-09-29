import cv2
import mediapipe as mp
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import os
from datetime import datetime
import argparse
import time
import glob
from pathlib import Path


class PoseExtractor:
    """
    A comprehensive pose extraction system for images and videos using MediaPipe.
    Extracts 33 pose landmarks with high accuracy and saves results in JSON format.
    """

    def __init__(self,
                 static_image_mode: bool = False,
                 model_complexity: int = 1,
                 smooth_landmarks: bool = True,
                 enable_segmentation: bool = False,
                 smooth_segmentation: bool = True,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize the PoseExtractor with MediaPipe configuration.

        Args:
            static_image_mode: Whether to treat input as static images
            model_complexity: Complexity of pose model (0, 1, or 2)
            smooth_landmarks: Whether to smooth landmarks across frames
            enable_segmentation: Whether to generate segmentation mask
            smooth_segmentation: Whether to smooth segmentation across frames
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
        """
        # Validate configuration parameters
        self._validate_configuration(
            model_complexity, min_detection_confidence, min_tracking_confidence
        )

        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            smooth_segmentation=smooth_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # Define landmark names for better JSON structure
        self.landmark_names = [
            'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
            'left_index', 'right_index', 'left_thumb', 'right_thumb',
            'left_hip', 'right_hip', 'left_knee', 'right_knee',
            'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index'
        ]

        # Configuration constants
        self.VISIBILITY_THRESHOLD = 0.3
        self.MAX_FRAMES_PROCESS = 10000  # Safety limit for large videos

    def _validate_configuration(self, model_complexity: int,
                                min_detection_confidence: float,
                                min_tracking_confidence: float) -> None:
        """Validate initialization parameters."""
        if model_complexity not in [0, 1, 2]:
            raise ValueError("model_complexity must be 0, 1, or 2")

        if not 0 <= min_detection_confidence <= 1:
            raise ValueError("min_detection_confidence must be between 0 and 1")

        if not 0 <= min_tracking_confidence <= 1:
            raise ValueError("min_tracking_confidence must be between 0 and 1")

    def _validate_image(self, image: np.ndarray, image_path: str) -> None:
        """Validate loaded image."""
        if image is None:
            raise FileNotFoundError(f"Image not found or unreadable: {image_path}")

        if image.size == 0:
            raise ValueError(f"Loaded image is empty: {image_path}")

        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"Image must be 3-channel color image: {image_path}")

    def _extract_landmarks(self, results) -> Optional[Dict]:
        """
        Extract pose landmarks from MediaPipe results.

        Args:
            results: MediaPipe pose detection results

        Returns:
            Dictionary containing landmark data or None if no pose detected
        """
        if not results.pose_landmarks:
            return None

        landmarks = {}
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            landmark_name = self.landmark_names[idx] if idx < len(self.landmark_names) else f'landmark_{idx}'
            landmarks[landmark_name] = {
                'x': float(landmark.x),
                'y': float(landmark.y),
                'z': float(landmark.z),
                'visibility': float(landmark.visibility)
            }

        return landmarks

    def _is_landmark_visible(self, landmark: Dict) -> bool:
        """Check if landmark meets visibility threshold."""
        return landmark['visibility'] > self.VISIBILITY_THRESHOLD

    def _calculate_pose_angles(self, landmarks: Dict) -> Dict:
        """
        Calculate key pose angles for biomechanical analysis.

        Args:
            landmarks: Dictionary of pose landmarks

        Returns:
            Dictionary containing calculated angles
        """

        def calculate_angle(p1: Dict, p2: Dict, p3: Dict) -> float:
            """Calculate angle between three points."""
            try:
                a = np.array([p1['x'], p1['y']])
                b = np.array([p2['x'], p2['y']])
                c = np.array([p3['x'], p3['y']])

                ba = a - b
                bc = c - b

                cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
                angle = np.arccos(cosine_angle)

                return float(np.degrees(angle))
            except (ValueError, ZeroDivisionError):
                return 0.0

        angles = {}

        try:
            # Left arm angle (shoulder-elbow-wrist)
            if (all(key in landmarks for key in ['left_shoulder', 'left_elbow', 'left_wrist']) and
                    all(self._is_landmark_visible(landmarks[key]) for key in
                        ['left_shoulder', 'left_elbow', 'left_wrist'])):
                angles['left_elbow_angle'] = calculate_angle(
                    landmarks['left_shoulder'],
                    landmarks['left_elbow'],
                    landmarks['left_wrist']
                )

            # Right arm angle
            if (all(key in landmarks for key in ['right_shoulder', 'right_elbow', 'right_wrist']) and
                    all(self._is_landmark_visible(landmarks[key]) for key in
                        ['right_shoulder', 'right_elbow', 'right_wrist'])):
                angles['right_elbow_angle'] = calculate_angle(
                    landmarks['right_shoulder'],
                    landmarks['right_elbow'],
                    landmarks['right_wrist']
                )

            # Left leg angle (hip-knee-ankle)
            if (all(key in landmarks for key in ['left_hip', 'left_knee', 'left_ankle']) and
                    all(self._is_landmark_visible(landmarks[key]) for key in ['left_hip', 'left_knee', 'left_ankle'])):
                angles['left_knee_angle'] = calculate_angle(
                    landmarks['left_hip'],
                    landmarks['left_knee'],
                    landmarks['left_ankle']
                )

            # Right leg angle
            if (all(key in landmarks for key in ['right_hip', 'right_knee', 'right_ankle']) and
                    all(self._is_landmark_visible(landmarks[key]) for key in
                        ['right_hip', 'right_knee', 'right_ankle'])):
                angles['right_knee_angle'] = calculate_angle(
                    landmarks['right_hip'],
                    landmarks['right_knee'],
                    landmarks['right_ankle']
                )

            # Torso angle (shoulder-hip alignment)
            if (all(key in landmarks for key in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']) and
                    all(self._is_landmark_visible(landmarks[key]) for key in
                        ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'])):
                left_torso = calculate_angle(
                    landmarks['left_shoulder'],
                    landmarks['left_hip'],
                    landmarks['right_hip']
                )
                right_torso = calculate_angle(
                    landmarks['right_shoulder'],
                    landmarks['right_hip'],
                    landmarks['left_hip']
                )
                angles['torso_lean'] = (left_torso + right_torso) / 2

        except Exception as e:
            print(f"Warning: Could not calculate some angles: {e}")

        return angles

    def extract_from_image(self, image_path: str) -> Dict:
        """
        Extract pose from a single image.

        Args:
            image_path: Path to the input image

        Returns:
            Dictionary containing pose data
        """
        start_time = time.time()

        try:
            image = cv2.imread(image_path)
            self._validate_image(image, image_path)

            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the image
            results = self.pose.process(rgb_image)

            # Extract landmarks
            landmarks = self._extract_landmarks(results)

            processing_time = time.time() - start_time

            pose_data = {
                'source': image_path,
                'source_type': 'image',
                'timestamp': datetime.now().isoformat(),
                'image_dimensions': {
                    'width': image.shape[1],
                    'height': image.shape[0],
                    'channels': image.shape[2]
                },
                'pose_detected': landmarks is not None,
                'landmarks': landmarks,
                'angles': self._calculate_pose_angles(landmarks) if landmarks else {},
                'confidence_score': self._calculate_overall_confidence(landmarks) if landmarks else 0.0,
                'processing_time_seconds': round(processing_time, 3),
                'error': None
            }

            return pose_data

        except Exception as e:
            return {
                'source': image_path,
                'source_type': 'image',
                'timestamp': datetime.now().isoformat(),
                'pose_detected': False,
                'error': str(e),
                'processing_time_seconds': round(time.time() - start_time, 3)
            }

    def extract_from_video(self, video_path: str, frame_interval: int = 1) -> Dict:
        """
        Extract pose from video frames.

        Args:
            video_path: Path to the input video
            frame_interval: Extract pose every N frames (1 = every frame)

        Returns:
            Dictionary containing pose data for all frames
        """
        start_time = time.time()
        cap = None

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            frames_data = []
            frame_count = 0
            frames_processed = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Safety limit for very large videos
                if frames_processed >= self.MAX_FRAMES_PROCESS:
                    print(f"Warning: Processed maximum of {self.MAX_FRAMES_PROCESS} frames")
                    break

                if frame_count % frame_interval == 0:
                    frame_start_time = time.time()

                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Process the frame
                    results = self.pose.process(rgb_frame)

                    # Extract landmarks
                    landmarks = self._extract_landmarks(results)

                    frame_data = {
                        'frame_number': frame_count,
                        'timestamp_seconds': round(frame_count / fps, 3) if fps > 0 else 0,
                        'pose_detected': landmarks is not None,
                        'landmarks': landmarks,
                        'angles': self._calculate_pose_angles(landmarks) if landmarks else {},
                        'confidence_score': self._calculate_overall_confidence(landmarks) if landmarks else 0.0,
                        'processing_time_seconds': round(time.time() - frame_start_time, 3)
                    }

                    frames_data.append(frame_data)
                    frames_processed += 1

                frame_count += 1

            total_processing_time = time.time() - start_time

            video_data = {
                'source': video_path,
                'source_type': 'video',
                'extraction_timestamp': datetime.now().isoformat(),
                'video_properties': {
                    'fps': fps,
                    'total_frames': total_frames,
                    'duration_seconds': round(duration, 3),
                    'width': width,
                    'height': height,
                    'frame_interval': frame_interval
                },
                'processing_metrics': {
                    'total_processing_time_seconds': round(total_processing_time, 3),
                    'frames_processed': len(frames_data),
                    'processing_fps': round(len(frames_data) / total_processing_time,
                                            2) if total_processing_time > 0 else 0
                },
                'frames_with_pose': sum(1 for f in frames_data if f['pose_detected']),
                'frames': frames_data,
                'error': None
            }

            return video_data

        except Exception as e:
            return {
                'source': video_path,
                'source_type': 'video',
                'extraction_timestamp': datetime.now().isoformat(),
                'error': str(e),
                'processing_time_seconds': round(time.time() - start_time, 3)
            }

        finally:
            if cap is not None:
                cap.release()

    def extract_from_directory(self, directory_path: str,
                               file_pattern: str = "*",
                               recursive: bool = False) -> Dict:
        """
        Process all images/videos in a directory.

        Args:
            directory_path: Path to directory containing media files
            file_pattern: File pattern to match (e.g., "*.jpg", "*.mp4")
            recursive: Whether to search subdirectories recursively

        Returns:
            Dictionary containing results for all processed files
        """
        start_time = time.time()

        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        # Supported file extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}

        # Find matching files
        pattern = os.path.join(directory_path, "**", file_pattern) if recursive else os.path.join(directory_path,
                                                                                                  file_pattern)
        files = glob.glob(pattern, recursive=recursive)

        results = {
            'batch_processing_timestamp': datetime.now().isoformat(),
            'directory_path': directory_path,
            'file_pattern': file_pattern,
            'recursive_search': recursive,
            'total_files_found': len(files),
            'processing_summary': {
                'images_processed': 0,
                'videos_processed': 0,
                'errors_encountered': 0
            },
            'files': []
        }

        for file_path in files:
            if not os.path.isfile(file_path):
                continue

            file_ext = os.path.splitext(file_path)[1].lower()

            try:
                if file_ext in image_extensions:
                    print(f"Processing image: {file_path}")
                    file_data = self.extract_from_image(file_path)
                    results['processing_summary']['images_processed'] += 1

                elif file_ext in video_extensions:
                    print(f"Processing video: {file_path}")
                    file_data = self.extract_from_video(file_path)
                    results['processing_summary']['videos_processed'] += 1

                else:
                    continue  # Skip unsupported files

                results['files'].append(file_data)

                if 'error' in file_data and file_data['error']:
                    results['processing_summary']['errors_encountered'] += 1

            except Exception as e:
                error_result = {
                    'source': file_path,
                    'error': str(e),
                    'pose_detected': False
                }
                results['files'].append(error_result)
                results['processing_summary']['errors_encountered'] += 1

        results['total_processing_time_seconds'] = round(time.time() - start_time, 3)
        return results

    def _calculate_overall_confidence(self, landmarks: Dict) -> float:
        """
        Calculate overall confidence score based on landmark visibility.

        Args:
            landmarks: Dictionary of pose landmarks

        Returns:
            Average confidence score (0.0 to 1.0)
        """
        if not landmarks:
            return 0.0

        visibilities = [landmark['visibility'] for landmark in landmarks.values()]
        return float(np.mean(visibilities))

    def save_to_json(self, data: Dict, output_path: str, indent: int = 2) -> None:
        """
        Save pose data to JSON file.

        Args:
            data: Pose data dictionary
            output_path: Output JSON file path
            indent: JSON formatting indent
        """
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)

        print(f"Pose data saved to: {output_path}")

    def visualize_pose(self, image_path: str, output_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize pose landmarks on image.

        Args:
            image_path: Path to input image
            output_path: Optional path to save visualization

        Returns:
            Image with pose landmarks drawn
        """
        try:
            image = cv2.imread(image_path)
            self._validate_image(image, image_path)

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = self.pose.process(rgb_image)

            # Draw pose landmarks
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )

            if output_path:
                cv2.imwrite(output_path, image)
                print(f"Visualization saved to: {output_path}")

            return image

        except Exception as e:
            print(f"Error creating visualization: {e}")
            raise

    def _print_summary(self, pose_data: Dict, is_batch: bool = False) -> None:
        """Print processing summary."""
        print(f"\n{'=' * 50}")
        print("PROCESSING SUMMARY")
        print(f"{'=' * 50}")

        if is_batch:
            print(f"Total files processed: {pose_data['total_files_found']}")
            print(f"Images processed: {pose_data['processing_summary']['images_processed']}")
            print(f"Videos processed: {pose_data['processing_summary']['videos_processed']}")
            print(f"Errors encountered: {pose_data['processing_summary']['errors_encountered']}")
            print(f"Total processing time: {pose_data.get('total_processing_time_seconds', 0):.2f}s")

        elif pose_data['source_type'] == 'image':
            detected = "Yes" if pose_data['pose_detected'] else "No"
            confidence = pose_data.get('confidence_score', 0)
            processing_time = pose_data.get('processing_time_seconds', 0)

            print(f"Pose detected: {detected}")
            if pose_data['pose_detected']:
                print(f"Confidence score: {confidence:.3f}")
            print(f"Processing time: {processing_time:.3f}s")

        else:  # Video
            frames_processed = len(pose_data.get('frames', []))
            frames_with_pose = pose_data.get('frames_with_pose', 0)
            processing_metrics = pose_data.get('processing_metrics', {})

            print(f"Frames processed: {frames_processed}")
            print(f"Frames with pose: {frames_with_pose}")
            if frames_processed > 0:
                print(f"Detection rate: {frames_with_pose / frames_processed * 100:.1f}%")
            print(f"Processing FPS: {processing_metrics.get('processing_fps', 0):.2f}")
            print(f"Total processing time: {processing_metrics.get('total_processing_time_seconds', 0):.2f}s")

        print(f"{'=' * 50}")


def main():
    """Main function to run pose extraction from command line."""
    parser = argparse.ArgumentParser(description='Extract pose from images or videos')
    parser.add_argument('input', help='Input image, video, or directory path')
    parser.add_argument('--output', '-o', help='Output JSON file path')
    parser.add_argument('--frame-interval', '-f', type=int, default=1,
                        help='Frame interval for video processing (default: 1)')
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Create pose visualization')
    parser.add_argument('--model-complexity', '-m', type=int, default=1, choices=[0, 1, 2],
                        help='Model complexity (0: fast, 1: balanced, 2: accurate)')
    parser.add_argument('--batch', '-b', action='store_true',
                        help='Process all media files in directory')
    parser.add_argument('--file-pattern', '-p', default='*',
                        help='File pattern for batch processing (e.g., "*.jpg")')
    parser.add_argument('--recursive', '-r', action='store_true',
                        help='Search subdirectories recursively in batch mode')

    args = parser.parse_args()

    # Initialize pose extractor
    extractor = PoseExtractor(model_complexity=args.model_complexity)

    try:
        if args.batch:
            # Process directory in batch mode
            print(f"Processing directory: {args.input}")
            pose_data = extractor.extract_from_directory(
                args.input,
                file_pattern=args.file_pattern,
                recursive=args.recursive
            )

        else:
            # Determine file type for single file processing
            if os.path.isdir(args.input):
                raise ValueError("Input is a directory. Use --batch flag for directory processing.")

            file_ext = os.path.splitext(args.input)[1].lower()
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}

            if file_ext in image_extensions:
                # Process image
                print(f"Processing image: {args.input}")
                pose_data = extractor.extract_from_image(args.input)

                # Create visualization if requested
                if args.visualize:
                    vis_path = f"{os.path.splitext(args.input)[0]}_pose_visualization.jpg"
                    extractor.visualize_pose(args.input, vis_path)

            elif file_ext in video_extensions:
                # Process video
                print(f"Processing video: {args.input}")
                pose_data = extractor.extract_from_video(args.input, args.frame_interval)

            else:
                raise ValueError(f"Unsupported file format: {file_ext}")

        # Save results
        if args.output:
            output_path = args.output
        else:
            if args.batch:
                base_name = os.path.basename(args.input.rstrip('/\\'))
                output_path = f"{base_name}_batch_pose_data.json"
            else:
                base_name = os.path.splitext(args.input)[0]
                output_path = f"{base_name}_pose_data.json"

        extractor.save_to_json(pose_data, output_path)

        # Print summ
        extractor._print_summary(pose_data, args.batch)

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())