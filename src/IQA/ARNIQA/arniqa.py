import torch
from PIL import Image
from torchvision import transforms
from IQA.ARNIQA.models.arniqa_model import ARNIQAModel
import gdown
import os
from dotenv import load_dotenv, find_dotenv


class ARNIQA:
    def __init__(self):
        load_dotenv(find_dotenv('.env'))
        model_filename = 'ARNIQA.pth'
        regressor_filename = 'regressor_koniq10k.pth'
        model_filedir = os.getcwd() + r'\IQA\ARNIQA\models\weights'
        self.__model_filepath = f"{model_filedir}\{model_filename}"
        self.__regressor_filepath = f"{model_filedir}\{regressor_filename}"

        if model_filename not in os.listdir(model_filedir):
            print("Wait download of ARNIQA Model")
            url_model = os.getenv('ARNIQA_MODEL_URL')
            gdown.download(url_model, self.__model_filepath, quiet=False, fuzzy=True)

        if not (regressor_filename in os.listdir(model_filedir)):
            print("Wait download of ARNIQA Regressor")
            url_regressor = os.getenv('ARNIQA_REGRESSOR_URL')
            gdown.download(url_regressor, self.__regressor_filepath, quiet=False, fuzzy=True)

        self.__device = torch.device('cuda') if torch.cuda.is_available() else "cpu"

        self.model = ARNIQAModel(self.__model_filepath, self.__regressor_filepath).to(self.__device)
        self.model.eval().to(self.__device)

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def clear_allocated_model(self):
        del self.model
        torch.cuda.empty_cache()

    def predict(self, image_path: str) -> float:
        img = Image.open(image_path).convert('RGB')

        # Get the half-scale image
        img_ds = transforms.Resize((img.size[1] // 2, img.size[0] // 2))(img)

        # Preprocess the images
        img = self.preprocess(img).unsqueeze(0).to(self.__device)
        img_ds = self.preprocess(img_ds).unsqueeze(0).to(self.__device)

        with torch.no_grad(), torch.amp.autocast("cuda"):
            score = self.model(img, img_ds, return_embedding=False, scale_score=True)

        return score.item()

    def find_issues(self, std_score, image_path):
        return std_score, "Not found"
