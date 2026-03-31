"""Face recognition tool — supports local OpenCV and cloud APIs."""

from typing import Literal


class FaceRecognition:
    """Face recognition tool."""

    def __init__(self, mode: Literal["local", "baidu"] = "local"):
        self.mode = mode

    def recognize(self, image_path: str) -> dict:
        """Recognize faces in an image."""
        if self.mode == "local":
            return self._recognize_local(image_path)
        elif self.mode == "baidu":
            return self._recognize_baidu(image_path)
        raise ValueError(f"Unknown mode: {self.mode}")

    def _recognize_local(self, image_path: str) -> dict:
        """Local face recognition using OpenCV/face_recognition."""
        # TODO: implement with OpenCV or face_recognition library
        raise NotImplementedError("Local face recognition not yet implemented")

    def _recognize_baidu(self, image_path: str) -> dict:
        """Baidu face API recognition."""
        # TODO: implement with Baidu Face API
        raise NotImplementedError("Baidu face recognition not yet implemented")
