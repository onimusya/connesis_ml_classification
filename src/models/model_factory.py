from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from src.utils.logger import handle_error, log_info


def get_default_hyperparameters():
    """Get default hyperparameters for all models"""
    return {
        "Decision Tree": {
            'max_depth': 5,
            'min_samples_split': 2,
            'min_samples_leaf': 1
        },
        "KNN": {
            'n_neighbors': 5,
            'weights': 'uniform',
            'metric': 'minkowski'
        },
        "Random Forest": {
            'n_estimators': 100,
            'max_depth': 5,
            'min_samples_split': 2
        },
        "SVM": {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale'
        },
        "Logistic Regression": {
            'C': 1.0,
            'solver': 'liblinear',
            'max_iter': 1000
        },
        "Gradient Boosting": {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3
        }
    }


def get_hyperparameter_ranges():
    """Get hyperparameter ranges for grid search"""
    return {
        "Decision Tree": {
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        "KNN": {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['minkowski', 'euclidean', 'manhattan']
        },
        "Random Forest": {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10]
        },
        "SVM": {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto']
        },
        "Logistic Regression": {
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs'],
            'max_iter': [1000, 2000]
        },
        "Gradient Boosting": {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    }


def create_model(model_name, hyperparams, class_weight=False):
    """Create model instance with specified hyperparameters"""
    try:
        class_weight_param = 'balanced' if class_weight else None
        
        if model_name == "Decision Tree":
            model = DecisionTreeClassifier(
                max_depth=hyperparams.get('max_depth'),
                min_samples_split=hyperparams.get('min_samples_split'),
                min_samples_leaf=hyperparams.get('min_samples_leaf', 1),
                class_weight=class_weight_param,
                random_state=42
            )
            
        elif model_name == "KNN":
            model = KNeighborsClassifier(
                n_neighbors=hyperparams.get('n_neighbors'),
                weights=hyperparams.get('weights'),
                metric=hyperparams.get('metric', 'minkowski')
            )
            
        elif model_name == "Random Forest":
            model = RandomForestClassifier(
                n_estimators=hyperparams.get('n_estimators'),
                max_depth=hyperparams.get('max_depth'),
                min_samples_split=hyperparams.get('min_samples_split', 2),
                class_weight=class_weight_param,
                random_state=42
            )
            
        elif model_name == "SVM":
            model = SVC(
                C=hyperparams.get('C'),
                kernel=hyperparams.get('kernel'),
                gamma=hyperparams.get('gamma', 'scale'),
                class_weight=class_weight_param,
                probability=True,
                random_state=42
            )
            
        elif model_name == "Logistic Regression":
            model = LogisticRegression(
                C=hyperparams.get('C'),
                solver=hyperparams.get('solver', 'liblinear'),
                max_iter=hyperparams.get('max_iter', 1000),
                class_weight=class_weight_param,
                random_state=42
            )
            
        elif model_name == "Gradient Boosting":
            model = GradientBoostingClassifier(
                n_estimators=hyperparams.get('n_estimators'),
                learning_rate=hyperparams.get('learning_rate'),
                max_depth=hyperparams.get('max_depth'),
                random_state=42
            )
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        log_info(f"Created {model_name} model with hyperparameters: {hyperparams}")
        return model
        
    except Exception as e:
        handle_error(e, f"Error creating {model_name} model")
        return None


def get_model_list():
    """Get list of available models"""
    return ["Decision Tree", "KNN", "Random Forest", "SVM", "Logistic Regression", "Gradient Boosting"]