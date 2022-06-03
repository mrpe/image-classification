import io
import torch.nn as nn
from torch.utils.data import dataloader
from ml.dataset import SingleByteData
from ml.model import Model
import torch


class Predictor:
    """
    The Predictor class is used for making predictions using a trained model instance based on the Model class
    defined in ml.model.py and the training steps defined in ml.trainer.py
    """

    def __init__(self):
        self.model = Model()
        self.model.load_model('models/2022-02-25-myModel.pt')

        self.softmax = nn.Softmax(dim=-1)

    async def predict(self, file):
        """
        Performs prediction on a sample using the model at the given path
        """

        torch.no_grad()

        # Preprocess the inputted sample to prepare it for the model
        preprocessed_sample = await self._preprocess(file)

        # Forward the preprocessed sample into the model as defined in the __call__ function in the Model class
        for img in preprocessed_sample:
            prediction = self.model(img)

        # Postprocess the prediction to prepare it for the client
        prediction = self._postprocess(prediction)

        return {'cat': str(prediction[0]), 'dog': str(prediction[1])}

    async def _preprocess(self, sample):
        bin_data = io.BytesIO(await sample.read())
        return dataloader.DataLoader(SingleByteData(bin_data), 4, True)

    def _postprocess(self, prediction):
        print(prediction)
        return self.softmax(prediction).tolist()[0]
