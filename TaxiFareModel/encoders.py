from sklearn.base import BaseEstimator, TransformerMixin
from utils import haversine_vectorized

class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
    """Extract the day of week (dow), the hour, the month and the year from a
    time column."""
    def __init__(self, time_column='pickup_datetime', time_zone_name='America/New_York'):
        self.time_column = time_column
        self.time_zone_name = time_zone_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Returns a copy of the DataFrame X with only four columns: 'dow', 'hour', 'month', 'year'"""
        X_c = X.copy()
        timezone_name = self.time_zone_name
        time_column = self.time_column
        X_c.index = pd.to_datetime(X_c[time_column])
        X_c.index = X_c.index.tz_convert(timezone_name)
        X_c["dow"] = X_c.index.weekday
        X_c["hour"] = X_c.index.hour
        X_c["month"] = X_c.index.month
        X_c["year"] = X_c.index.year
        X_c.reset_index(drop=True)
        return X_c[['dow','hour','month', 'year']]


class DistanceTransformer(BaseEstimator, TransformerMixin):
    """Compute the haversine distance between two GPS points."""
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Returns a copy of the DataFrame X with only one column: 'distance'"""
        X_c = X.copy()
        X_c['distance'] = haversine_vectorized(X)
        return X_c[['distance']]

