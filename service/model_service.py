from tensorflow import get_default_graph
from tensorflow.keras.backend import set_session
from tensorflow.keras.models import load_model

from tensorflow import Session

SESS = Session()
GRAPH = get_default_graph()
set_session(SESS)
MODEL = load_model('/root/api/res/model_10.h5')

class ModelService:
    def predict(self, image):
        global sess
        global graph

        with GRAPH.as_default():
            set_session(SESS)
            preds = MODEL.predict(image)
            preds = [round(i*100) for i in preds[0]]
            print(preds)

        return preds
