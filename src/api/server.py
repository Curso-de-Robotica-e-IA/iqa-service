"""This module contains the server configuration and the start_server function.
The start_server function starts the server with the specified host and port
in the .env file."""

import os

from flask import Flask
from dotenv import load_dotenv
from api.controllers.iqa_blueprint import iqa_controller
from api.controllers.root_controller import root_controller


app = Flask(__name__)


app.register_blueprint(root_controller, url_prefix='/')
app.register_blueprint(iqa_controller, url_prefix='/iqa')


def start_server():
    """Starts the server with the specified host and port in the .env file.
    Additionally, the debug mode is enabled if specified in the .env file.
    """
    load_dotenv()
    host = os.getenv('HOST')
    port = int(os.getenv('PORT'))
    debug = True if int(os.getenv('DEBUG')) == 1 else False
    app.run(host=host, port=port, debug=debug)
