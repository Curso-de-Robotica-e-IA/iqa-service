from IQA.DIQA.diqa import DIQA


class IQA:
    def __init__(self):
        self.__version = 1
        self.__api = DIQA()

    def version_info(self) -> int:
        return self.__version

    def predict(self, image_path) -> float:
        return self.__api.predict(image_path)

    def find_issues(self, std_score, image_path) -> tuple:
        return self.__api.find_issues(std_score, image_path)

    def clear(self):
        return self.__api.clear_allocated_model()
