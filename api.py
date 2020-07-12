import flask
from service.model_service import ModelService
from util.image_utils import *
from util.constants import LABELS

APP = flask.Flask(__name__)
MODEL_SERVICE = ModelService()

@APP.route('/predict', methods=['POST'])
def predict():
    print('INICIO predict')
    data = {'success': False}

    if flask.request.method == 'POST':
        if flask.request.files.get('image'):
            image = flask.request.files['image'].read()
            image = map_format(image)
            image = prepare_image(image, target=(96, 96))
            preds = MODEL_SERVICE.predict(image)

            data['predictions'] = []

            for i in range(len(preds)):
                response = {'label': LABELS[i], 'prob': float(preds[i])}
                data['predictions'].append(response)
            data['success'] = True

    print('FIN predict')
    return flask.jsonify(data)

if __name__ == '__main__':
    print('- SkinCareAI-API -')
    APP.run(host='0.0.0.0')
