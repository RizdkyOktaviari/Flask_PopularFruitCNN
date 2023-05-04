from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from werkzeug.utils import secure_filename
import os
from pred import prediction


UPLOAD_FOLDER = '/home/ubuntu/Flask_PopularFruitCNN/static/uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/upload", methods=["POST"])
def upload():
    if 'file' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
    file = request.files['file']
    if file.filename == '':
        resp = jsonify({'message' : 'No file selected for uploading'})
        resp.status_code = 400
        return resp
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        prediction_result = prediction(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        resp = jsonify(
        {
            'url_image': os.path.join(app.config['UPLOAD_FOLDER'], filename),
            'stage' : str(prediction_result),
            'message' : 'File successfully uploaded'        
        }
        )
        resp.status_code = 201
        return resp
    else:
        resp = jsonify({'message' : 'Allowed file types are png, jpg, jpeg, gif'})
        resp.status_code = 400
        return resp

if __name__ == "__main__":
    app.run()