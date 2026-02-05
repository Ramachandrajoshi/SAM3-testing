import cv2
import os

class VideoHandler:
    """
    Utility class for reading and writing video files/streams.
    """
    
    @staticmethod
    def get_video_properties(video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return fps, width, height, total_frames

    @staticmethod
    def create_video_writer(output_path, fps, width, height):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    @staticmethod
    def process_camera(camera_index=0, callback=None):
        """
        Stream from camera and apply callback for each frame.
        """
        cap = cv2.VideoCapture(camera_index)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if callback:
                frame = callback(frame)
            
            cv2.imshow('NSP Visual Analysis System Real-time Analysis', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
