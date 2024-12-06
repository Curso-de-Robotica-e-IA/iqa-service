"""Blueprint for IQA controller. The IQA controller defines the routes for the
Image Quality Assessment (IQA) API.
"""

from flask import Blueprint, request, Response, jsonify
from http import HTTPStatus
from api.services.iqa_service import IQAService


iqa_controller = Blueprint('iqa_controller', __name__)


@iqa_controller.route('/evaluate', methods=['POST'])
def evaluate() -> Response:
    """Evaluates the image quality of an image. The image is sent as a base64
    string in the request body. The result is a JSON object containing the
    score and the comment of the image quality assessment.

    The request body must contain the following fields:
    - data (str): The base64 string of the image to be evaluated.

    Returns:
        Response: The response containing the result of the image quality
        assessment.
    """
    try:
        body = request.json
        iqa_service = IQAService()
        img = iqa_service._img_from_base64(body.get('data'))
        iqa_result = iqa_service.evaluate(img)
        res = jsonify(iqa_result)
        res.status_code = HTTPStatus.OK
        return res
    except Exception as e:
        print(e)
        return Response(status=HTTPStatus.INTERNAL_SERVER_ERROR)
