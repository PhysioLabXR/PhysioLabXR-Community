from abc import ABC, abstractmethod

class RealTimeModel(ABC):
    """
    An abstract class for implementing scripting models.
    """
    expected_input_size = None
    expected_preprocessed_input_size = None

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, new_model):
        self.__model = new_model

    @property
    def data_min(self):
        return self.__data_min

    @data_min.setter
    def data_min(self, value):
        self.__data_min = value

    @property
    def data_max(self):
        return self.__data_max

    @data_max.setter
    def data_max(self, value):
        self.__data_max = value

    # @abstractmethod
    # def resample(self, input, freq):
    #     """
    #     Implement for model-specific resampling method.
    #     This method will be used in preprocess().
    #     """
    #     pass

    @abstractmethod
    def preprocess(self, input):
        return input
        # if input.shape != self.expected_input_size:
        #     raise ValueError("Unexpected Input Size Provided.")
        #
        # # resample
        # preprocessed = self.resample(input, __)
        #
        # # min max normalization
        # preprocessed = (preprocessed - self.data_min) / (self.data_max - self.data_min)
        #
        # # slice into samples
        #
        # return preprocessed

    def predict(self, input, **kwargs):
        """
        Have ML Model predict the probability of the data being a target,
        a distractor, or a novelty.

        Returns 3 probability values.
        """
        if input.shape[1:] != self.expected_preprocessed_input_size:
            raise ValueError("Unexpected Expected Input Size Provided.")

        y_hypo = self.model.predict(input)
        return y_hypo

    def prepare_model(self, model_path, preprocess_params_path):
        self.model = tf.keras.models.load_model(model_path)
        # self.data_min, self.data_max = load_params(preprocess_params_path)
