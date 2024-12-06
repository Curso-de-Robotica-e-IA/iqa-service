import os
import base64
import cv2
import numpy as np
import tempfile
from IQA.iqa import IQA
from typing import TypedDict


class IQAResult(TypedDict):
    """Represents the result of an IQA evaluation.
    TypedDict (dict): A dictionary with the following keys:

    Args:
        score (float): The score of the image quality assessment.
        comment (str): The comment of the image quality assessment.
    """
    score: float
    comment: str


class IQAService:
    """Represents the service for the Image Quality Assessment (IQA) API.
    """

    def __init__(self):
        """Initializes the IQA service. The model used for the IQA evaluation
        is chosen at the IQA class, from the `.env` file values.
        """
        self.__iqa = IQA()

    @staticmethod
    def _img_from_base64(base64_str: str) -> bytes:
        """Converts a base64 string to an image.

        Args:
            base64_str (str): The base64 string to be converted.

        Returns:
            bytes: The image converted from the base64 string.
        """
        img = base64.b64decode(base64_str)
        nparr = np.frombuffer(img, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_bytes = cv2.imencode('.jpg', img_np)[1].tobytes()
        return img_bytes

    def evaluate(self, img_data: bytes) -> IQAResult:
        """Evaluates the image quality of an image. The image is saved in a
        temporary file, and the IQA model predicts the image quality. The
        result is a dictionary containing the score and the comment of the
        image quality assessment.

        Args:
            img_data (bytes): The image data to be evaluated.

        Returns:
            IQAResult: The result of the image quality assessment.
        """
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(img_data)
        temp_file.close()

        try:
            prediction = self.__iqa.predict(temp_file.name)
            _, issues = self.__iqa.find_issues(prediction, temp_file.name)
        finally:
            os.remove(temp_file.name)

        result = IQAResult(score=prediction, comment=issues)
        return result
