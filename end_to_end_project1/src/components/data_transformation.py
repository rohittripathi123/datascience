from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer



from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import joblib
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.logger import logger  # use your logger

def transform_data(df):
    try:
        logger.info("Starting data transformation...")

        # Split features and target
        X = df.drop(columns=['math_score'], axis=1)
        y = df['math_score']

        logger.info(f"Shape of dataset: X={X.shape}, y={y.shape}")

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        logger.info(f"Data split into train={X_train.shape}, test={X_test.shape}")

        # Define column types
        num_features = X_train.select_dtypes(exclude="object").columns
        cat_features = X_train.select_dtypes(include="object").columns

        logger.info(f"Numerical features: {list(num_features)}")
        logger.info(f"Categorical features: {list(cat_features)}")

        # Transformers
        numeric_transformer = StandardScaler()
        oh_transformer = OneHotEncoder(handle_unknown="ignore")  

        # Column transformer
        preprocessor = ColumnTransformer(
            [
                ("OneHotEncoder", oh_transformer, cat_features),
                ("StandardScaler", numeric_transformer, num_features),
            ]
        )

        # Fit on training data
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)

        # Save preprocessor
        joblib.dump(preprocessor, "preprocessor.pkl")
        logger.info("Preprocessor saved as preprocessor.pkl")

        return X_train, X_test, y_train, y_test, preprocessor

    except Exception as e:
        logger.error("Error in transform_data", exc_info=True)
        raise e
