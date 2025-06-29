import unittest
from src.training.trainer import Trainer
from src.training.experiments import log_experiment, load_experiment

class TestTrainer(unittest.TestCase):

    def setUp(self):
        self.trainer = Trainer()
        # Additional setup can be done here

    def test_training_process(self):
        # Test the training process
        result = self.trainer.train()
        self.assertTrue(result['success'], "Training should be successful")

    def test_validation_process(self):
        # Test the validation process
        validation_result = self.trainer.validate()
        self.assertTrue(validation_result['success'], "Validation should be successful")

    def test_checkpoint_saving(self):
        # Test if checkpoints are saved correctly
        checkpoint_path = self.trainer.save_checkpoint()
        self.assertTrue(checkpoint_path.exists(), "Checkpoint should be saved")

class TestExperimentManagement(unittest.TestCase):

    def test_log_experiment(self):
        # Test logging an experiment
        experiment_id = log_experiment("Test Experiment", {"param": 1})
        self.assertIsNotNone(experiment_id, "Experiment ID should not be None")

    def test_load_experiment(self):
        # Test loading an experiment
        experiment_data = load_experiment("Test Experiment")
        self.assertIsNotNone(experiment_data, "Loaded experiment data should not be None")

if __name__ == '__main__':
    unittest.main()