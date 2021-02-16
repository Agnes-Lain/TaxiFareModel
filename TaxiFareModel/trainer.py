from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
# from utils import compute_rmse
from TaxiFareModel.utils import compute_rmse
import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property
from TaxifareModel.data import get_data, clean_data

MLFLOW_URI = "https://mlflow.lewagon.co/"
myname = "zuzu"
EXPERIMENT_NAME = f"TaxifareModel_{myname}"

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
        base_pipe = make_pipeline(preprocessor,
                                  model)
        self.pipeline = base_pipe

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        trained = self.pipeline.fit(self.X, self.y)
        return trained

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.run().predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        self.mlflow_log_param('model', "LinearRegression")
        self.mlflow_log_param('rmse', rmse)
        return rmse


    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":
    # get_data()
    df = get_data()
    # clean_data
    df = clean_data(df)
    # set X and y
    # hold out
    # train
    # evaluate
    print(df.head(3))
