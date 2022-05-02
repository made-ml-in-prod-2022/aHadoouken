import unittest
import os
import pandas as pd
import numpy as np

from src.main import train_model_pipeline, predict_model_pipeline
from src.data import obtain_data, load_data
from src.utils import DataParams, DownloadParams, SplittingParams
from src.features import DataTransformer


class TestML(unittest.TestCase):
    def test_training_pipeline_lr(self):
        config = os.path.abspath("tests/configs/config_lr_test.yaml")
        train_model_pipeline(config)

    def test_training_pipeline_lr(self):
        config = os.path.abspath("tests/configs/config_rf_test.yaml")
        train_model_pipeline(config)

    def test_predict_pipeline(self):
        model = os.path.abspath("tests/models/model.pkl")
        data = os.path.abspath("tests/data/data.csv")
        results = os.path.abspath("tests/data/res.csv")
        predict_model_pipeline(model, data, results)

    def test_data(self):
        with self.assertRaises(IOError):
            load_data("some/random/path")

        test_df = pd.DataFrame(
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
            columns=["a", "b", "c"],
        )
        test_df.to_csv("tests/data/test.csv", index=False)
        loaded_df = load_data("tests/data/test.csv")
        self.assertTrue(np.all(test_df == loaded_df))
        params = DataParams(
            "tests/data/test.csv",
            DownloadParams(False, "str"),
            SplittingParams(test_size=0.5),
        )
        df_train, df_test = obtain_data(params)
        self.assertEqual(len(df_test.merge(test_df)), len(df_test))
        self.assertEqual(len(df_train.merge(test_df)), len(df_train))
        self.assertEqual(
            (df_test.reset_index() == df_train.reset_index()).sum().sum(), 0
        )

    def test_transformer(self):
        test_df = pd.DataFrame(
            np.array([[1.1, 2, 1], [4.4, 5, 2], [0, 8, 2], [-10, 11, 1]]),
            columns=["num1", "num2", "cat"],
        )
        ref_arr =  np.array(
            [[1, 0, 0.35857958, -1.161895],
            [0, 1, 0.89040547, -0.38729833],
            [0, 1, 0.18130428, 0.38729833],
            [1, 0, -1.43028932, 1.161895]])
        cat_features = ["cat"]
        num_features = ["num1", "num2"]
        data_transformer = DataTransformer(cat_features, num_features)
        transformed_arr = data_transformer.fit_transform(test_df)
        self.assertTrue(np.allclose(transformed_arr, ref_arr))


if __name__ == "__main__":
    unittest.main()
