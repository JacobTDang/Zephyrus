import cv2, os, sys, time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import BaseOptions, vision
from mediapipe.tasks.python.vision import HandLandmarkerResult

LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]


class Vision:

    def __init__(self):
        self.path_to_model = os.path.join(os.path.dirname(__file__), '..', 'hand_landmarker_f16.task')
        self.latest_result = None
        self._setup_tasks()
        self._set_up_camera()

    def print_result(self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.latest_result = result

    def _draw_landmarks(self, frame, result: HandLandmarkerResult):
        h, w = frame.shape[:2]
        for hand_landmarks in result.hand_landmarks:
            for i, lm in enumerate(hand_landmarks):
                # normallize the image
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(frame, LANDMARK_NAMES[i], (x + 6, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    def _set_up_camera(self):
        self.capture = cv2.VideoCapture(0)

        with self.hand_landmarker.create_from_options(self.options) as landmarker:
            while self.capture.isOpened():
                success, frame = self.capture.read()
                if not success:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                timestamp_ms = int(time.time() * 1000)
                landmarker.detect_async(mp_image, timestamp_ms)

                if self.latest_result:
                    self._draw_landmarks(frame, self.latest_result)

                cv2.imshow("Hand Landmarks", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.capture.release()
        cv2.destroyAllWindows()

    def _setup_tasks(self):
        self.hand_landmarker = mp.tasks.vision.HandLandmarker
        self.hand_landmarker_options = mp.tasks.vision.HandLandmarkerOptions

        self.options = self.hand_landmarker_options(
            base_options=BaseOptions(model_asset_path=self.path_to_model),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            result_callback=self.print_result,
        )


if __name__ == "__main__":
    v = Vision()
