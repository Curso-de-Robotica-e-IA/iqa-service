from flask import Blueprint, Response
from http import HTTPStatus


root_controller = Blueprint('root_controller', __name__)


@root_controller.route('/', methods=['GET'])
def index():
    return Response(status=HTTPStatus.OK)


@root_controller.route('/help', methods=['GET'])
def help():
    return Response(status=HTTPStatus.NOT_IMPLEMENTED)
