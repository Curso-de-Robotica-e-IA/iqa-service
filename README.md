# IQA Service
This is a simple service that provides an API to calculate the Image Quality Assessment (IQA) of an image. The service is built using Flask and the IQA is calculated using the DIQA model.

## Installation
To install the service, you need anaconda or miniconda installed on your machine.
Once cloned the repository, you can create the environment using the following command:
```powershell
conda env create -f environment.yml
```
Then, you can activate the environment using:
```powershell
conda activate iqa-service
```
### .env file
The service uses a `.env` file to load the environment variables. You can create a `.env` file in the root of the project following the keys present in the `sample.env` file.

## Running the service
To run the service, you can use the following command:
```powershell
python .\src\main.py
```

## API
The service provides the following endpoints:

### GET /
This endpoint is used to check if the service is running. It returns a simple HTTP 200 response.

### GET /iqa/version
This endpoint is used to get the version of the DIQA model used by the service. It returns a JSON object with the version of the model.

```json
{
    "version": "1"
}
```

### POST /iqa/evaluate
This endpoint is used to evaluate the IQA of an image. The image should be sent as base64 string in the request body. The response is a JSON object with the IQA results and comments.

#### Request
```json
{
    "data": "base64_string"
}
```

#### Response
```json
{
    "comments": "The image is of good quality.",
    "score": 0.86584
}
```
