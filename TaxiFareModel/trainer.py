from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
# from utils import compute_rmse
from TaxiFareModel.utils import compute_rmse

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

        base_pipe = make_pipeline(preprocessor,
                                  LinearRegression())
        return base_pipe

    def run(self):
        """set and train the pipeline"""
        pipe = set_pipeline()
        trained = pipe.fit(X, y)
        return trained

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = run().predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse


if __name__ == "__main__":
    # get data
    # clean data
    # set X and y
    # hold out
    # train
    # evaluate
    print('TODO')
