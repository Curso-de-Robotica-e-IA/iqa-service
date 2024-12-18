from IQA.DIQA.diqa import DIQA


class IQA:
    """Represents the Image Quality Assessment (IQA) API.
    """
    def __init__(self):
        """Initializes the IQA API.
        """
        # TODO: choose a model for the IQA API using the `.env` file
        self.__version = 1
        self.__api = DIQA()

    def version_info(self) -> int:
        """Returns the version of the IQA API.

        Returns:
            int: The version of the IQA API.
        """
        return self.__version

    def predict(self, image_path: str) -> float:
        """Predicts the image quality of an image.

        Args:
            image_path (str): The path to the image to be evaluated.

        Returns:
            float: The predicted image quality score.
        """
        return self.__api.predict(image_path)

    def find_issues(self, std_score: float, image_path: str) -> tuple:
        """Finds the issues in an image.

        Args:
            std_score (float): The standard score of the image.
            image_path (str): The path to the image to be evaluated.

        Returns:
            tuple: A tuple containing the predicted score and the issues found.
        """
        return self.__api.find_issues(std_score, image_path)

    def clear(self):
        """Clears the allocated model.
        """
        self.__api.clear_allocated_model()
