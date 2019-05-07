import numpy


class SmartTrainer:
    def __init__(self):
        pass

    # Now that everything is connected it would suck to go through all the steps and have a model that isn't trained
    # So great.  This is my temporary super easy solution.
    def smart_train(self, input_matrix, output_matrix, model, validation_input_matrix, validation_output_matrix, hyper_params, training_function):
        num_times_through = 0
        trained_model = None
        while num_times_through < 50 and not self.should_stop_based_on_stats(model, hyper_params):
            print("We have not met the requirements to stop training.  We are trying another iteration! \n")
            print("The iteration number is: " + str(num_times_through) + "\n")
            trained_model = training_function(input_matrix, output_matrix, model, validation_input_matrix, validation_output_matrix, hyper_params)
            num_times_through += 1
        return trained_model

    def should_stop_based_on_stats(self, model, hyper_params):
        if not hasattr(model, 'history'):
            return False
        loss_history_mat = model.history.history['loss']
        if self.should_stop_based_on_loss_threshold(loss_history_mat, hyper_params) or \
                self.should_stop_because_mean_is_not_decreasing(loss_history_mat):
            return True
        else:
            return False

    @staticmethod
    def should_stop_based_on_loss_threshold(loss_history_mat, hyper_params):
        if loss_history_mat[-1] < hyper_params.training_loss_threshold:
            print("The model reached it's target loss! :) \n")
            return True
        else:
            return False

    @staticmethod
    def should_stop_because_mean_is_not_decreasing(loss_history_mat):
        num_to_split_into = 2
        tol = .00001
        split_loss_history = numpy.split(numpy.asarray(loss_history_mat), num_to_split_into)
        mean_first_half = numpy.mean(split_loss_history[0])
        mean_second_half = numpy.mean(split_loss_history[1])
        if mean_first_half < mean_second_half:
            print(
                "The mean loss for the first half versus the second half of this training matrix is greater... Were calling it a day \n")
            return True
        if -tol < mean_second_half - mean_first_half < tol:
            print("The mean from the first half was not different enough to keep training \n")
            return True
        return False
