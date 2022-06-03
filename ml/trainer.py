import torch.optim as optim
import torch.nn as nn
import torch.utils.data.dataloader as dataloader
from ml.model import Model
from ml.dataset import MyDataset
from tqdm import tqdm


class Trainer:
    """
    The Trainer class is used for training a model instance based on the Model class found in ml.model.py.
    In order to get started with training a model the following steps needs to be taken:
    1. Define the Model class in ml.model.py
    2. Prepare train data on which the model should be trained with by implementing the _read_train_data() function and
    the _preprocess_train_data() function
    """

    def __init__(self):
        # creates an instance of the Model class (see guidelines in ml.model.py)
        self.model = Model()
        self.model.train()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=0.001,
            momentum=0.9
        )
        self.softmax = nn.Softmax(dim=-1)

        self.criterion = nn.CrossEntropyLoss()

    def train(self, request):
        """
        Starts the training of a model based on data loaded by the self._load_train_data function
        """

        # Unpack request
        dataset_path = request.dataset_path
        save_path = request.save_path

        # Read the dataset from the dataset_path
        train_data = self._load_train_data()
        val_data = self._load_val_data()

        # Preprocess the dataset
        preprocessed_train_data = self._preprocess_train_data(train_data)

        num_epoch = 20
        for epoch in range(num_epoch):
            print('-------------------------------------------------------------------')
            print('Epoch ' + str(epoch + 1) + "/" + str(num_epoch))
            self.model.train()
            epoch_loss = 0
            batch_counter = 0

            for img, target in tqdm(train_data):
                self.optimizer.zero_grad()

                output = self.model(img)
                loss = self.criterion(output, target)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.step()

                epoch_loss += loss.item()
                batch_counter += 1

            print("Epoch loss: " + str(epoch_loss / batch_counter))

            correct_guesses = self._validate(val_data)
            print("% of correct guesses: " + str(correct_guesses))

            # Save the trained model
        return self.model.save_model(save_path)

    def _load_train_data(self):
        return dataloader.DataLoader(
            MyDataset("data/cleaned_train_data.txt"), 4, True, collate_fn=self.collate_fn
        )

    def _load_val_data(self):
        return dataloader.DataLoader(
            MyDataset("data/cleaned_test_data.txt"), 1, collate_fn=self.collate_fn
        )

    @ staticmethod
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return dataloader.default_collate(batch)

    def _validate(self, data_loader):
        print("Evaluating...")
        self.model.eval()
        correct_predictions = 0
        conf_threshold = 0.7
        sample_counter = 0
        for img, target in tqdm(data_loader):
            result = self.model(img)

            confidence = self.softmax(result).tolist()[0]
#            print("confidence:" +
#                  "[ " + str(confidence[0]) + " , " + str(confidence[1]) + " ]")
            max_value = max(confidence)
            max_index = confidence.index(max_value)

#            print("max value:" + str(max_value))
#            print("max index:" + str(max_index))

            if (max_value > conf_threshold and max_index == target.item()):
                # print("correct prediction!")
                # print(str(max_value) + ' > ' + str(conf_threshold))
                # print(str(max_index) + ' == ' + str(target.item()))
                correct_predictions += 1
            sample_counter += 1

        return correct_predictions / sample_counter

    def _preprocess_train_data(self, train_data):
        """
        TODO 2.: Implement preprocessing steps which prepares the dataset for training
        e.g. normalizing the data, removing noisy data, splitting up the data into input values and target values
        """
        preprocessed_train_data = train_data
        return preprocessed_train_data

    def __call__(self, request):
        return self.train(request)