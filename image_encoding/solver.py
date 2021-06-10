from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import numpy as np
from mlpm.solver import Solver

class EncodingSolver(Solver):
    def __init__(self, toml_file=None):
        super().__init__(toml_file)
        # Do you Init Work here
        base_model = VGG16(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
        self.ready()
    def infer(self, data):
        # if you need to get file uploaded, get the path from input_file_path in data
        img = Image.open(data['input_file_path'])
        img = img.resize((224,224))
        img = img.convert('RGB')
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = self.model.predict(x)[0]
        feature = feature / np.linalg.norm(feature)
        feature = feature.tolist()
        result = {'feature': str(feature)}
        return result # return a dict
