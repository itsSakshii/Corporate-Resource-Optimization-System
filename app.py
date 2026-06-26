"""Root-level entry point for the Corporate Resource Optimization Flask API.

This module imports the Flask application from the `app` package and runs it
on 0.0.0.0:5002 so that the service is reachable from outside the container.
Run `python main.py` first to train and persist the model artefacts, then
start this server with `python app.py` (or let the process manager do it).
"""

from app.app import app

if __name__ == '__main__':
    print("🚀 Corporate Resource Optimization API starting on 0.0.0.0:5002 ...")
    app.run(host='0.0.0.0', port=5002, debug=False)
