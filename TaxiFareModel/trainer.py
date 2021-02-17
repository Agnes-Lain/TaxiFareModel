from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
# from utils import compute_rmse
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property
from TaxiFareModel.data import get_data, clean_data, set_data
import joblib
from google.cloud import storage

MLFLOW_URI = "https://mlflow.lewagon.co/"
myname = "zuzu"
EXPERIMENT_NAME = f"TaxifareModel_{myname}"

BUCKET_NAME = 'wagon-ml-zong-project-01'

MODEL_NAME = 'taxifare'
MODEL_VERSION = 'v1'
FILENAME = 'pipeline_model.joblib'

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = make_pipeline(DistanceTransformer(),
                                  StandardScaler()
                                 )
        time_pipe = make_pipeline(TimeFeaturesEncoder(),
                                  OneHotEncoder()
                                 )
        preprocessor = ColumnTransformer([
                                    ('dist_transformer', dist_pipe, ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']),
                                    ('time_transformer', time_pipe, ['pickup_datetime'])])
        model = LinearRegression()
        self.pipeline = make_pipeline(preprocessor,
                                  model)
        return self.pipeline

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        return self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.run().predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        # self.mlflow_log_param('model', "LinearRegression")
        # self.mlflow_log_param('rmse', rmse)
        return rmse

    def save_model(self):
        """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
        HINTS : use joblib library and google-cloud-storage"""

        # saving the trained model to disk is mandatory to then beeing able to upload it to storage
        # Implement here
        print("saved model.joblib locally")
        trained_model = self.pipeline
        joblib.dump(trained_model, FILENAME)

        # Implement here
        storage_client = storage.Client()

        bucket = storage_client.bucket(BUCKET_NAME)

        storage_location = f'models/v2/{FILENAME}'

        blob = bucket.blob(storage_location)

        blob.upload_from_filename(FILENAME)

        print("uploaded model.joblib to gcp cloud storage under \n => {}".format(storage_location))



    # @memoized_property
    # def mlflow_client(self):
    #     mlflow.set_tracking_uri(MLFLOW_URI)
    #     return MlflowClient()

    # @memoized_property
    # def mlflow_experiment_id(self):
    #     try:
    #         return self.mlflow_client.create_experiment(self.experiment_name)
    #     except BaseException:
    #         return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    # @memoized_property
    # def mlflow_run(self):
    #     return self.mlflow_client.create_run(self.mlflow_experiment_id)

    # def mlflow_log_param(self, key, value):
    #     self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    # def mlflow_log_metric(self, key, value):
    #     self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":
    # get_data()
    df = get_data()
    # clean_data
    df = clean_data(df)
    # set X and y
    # hold out
    X_train, X_test, y_train, y_test = set_data(df)
    # train
    trained = Trainer(X_train, y_train)
    trained.run()
    # evaluate
    score = trained.evaluate(X_test, y_test)
    trained.save_model()
    print(score)
