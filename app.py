from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification, make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import logging
import random

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/api/model', methods=['POST'])
def model():
    try:
        data = request.json
        model_type = data['model_type']
        param = data['param']
        weights = data.get('weights', 'uniform')
        metric = data.get('metric', 'minkowski')
        kernel = data.get('kernel', 'linear')
        degree = data.get('degree', 3)
        sigma = data.get('sigma', 0.1)
        iteration_steps = data.get('iteration_steps', 1)
        random_data = data['randomData']

        logging.debug(f"Model: {model_type}, Param: {param}, Kernel: {kernel}, Degree: {degree}, Sigma: {sigma}, Iteration Steps: {iteration_steps}")

        X = np.array(random_data['X'])
        y = np.array(random_data['y'])

        model, y = train_model(model_type, X, y, param, kernel, degree, sigma, iteration_steps, weights, metric)
        if model is None:
            return jsonify({'error': 'Failed to train model'}), 500

        plot_url = create_plot(model, X, y, model_type)
        if plot_url:
            logging.debug(f"Plot URL length: {len(plot_url)}")
            return jsonify({'plot_url': plot_url})
        else:
            return jsonify({'error': 'Failed to generate plot'}), 500
    except Exception as e:
        logging.error(f"Error in model API endpoint: {e}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/api/randomize', methods=['POST'])
def randomize():
    try:
        data = request.json
        model_type = data['model_type']
        randomize_type = data.get('randomize_type', 'normal')  # Default to 'normal' if not provided
        
        logging.debug(f"Randomizing data for model type: {model_type} with randomize type: {randomize_type}")
        random_data = generate_random_data(model_type, randomize_type)  # Pass randomize type to the generator
        logging.debug(f"Generated random data: {random_data}")
        return jsonify(random_data)
    except Exception as e:
        logging.error(f"Error in randomize API endpoint: {e}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

# CSV-specific route and functionality
@app.route('/api/upload_csv', methods=['POST'])
def upload_csv():
    try:
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No file provided'}), 400

        df = pd.read_csv(file)
        if df.shape[1] != 3:
            return jsonify({'error': 'CSV must have exactly two features and one label column'}), 400

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1].values

        # Check for non-numerical columns in X
        non_numerical_columns = X.select_dtypes(exclude=['number']).columns

        if len(non_numerical_columns) > 0:
            # Apply LabelEncoder to non-numerical columns
            label_encoders = {}
            for col in non_numerical_columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                label_encoders[col] = le

        # Ensure X has exactly two features after encoding
        if X.shape[1] != 2:
            return jsonify({'error': 'After encoding categorical variables, there must be exactly two features'}), 400

        X = X.values  # Convert DataFrame back to NumPy array

        logging.debug(f"Uploaded CSV data: X={X}, y={y}")

        plot_url = create_csv_plot(X, y)
        if plot_url:
            return jsonify({'plot_url': plot_url, 'X': X.tolist(), 'y': y.tolist()})
        else:
            return jsonify({'error': 'Failed to generate plot from CSV data'}), 500
    except Exception as e:
        logging.error(f"Error processing CSV: {e}")
        return jsonify({'error': f'Failed to process the uploaded CSV file: {str(e)}'}), 500


@app.route('/api/model_csv', methods=['POST'])
def model_csv():
    try:
        data = request.json
        model_type = data['model_type']
        param = data.get('param', 1)
        weights = data.get('weights', 'uniform')
        metric = data.get('metric', 'minkowski')
        kernel = data.get('kernel', 'linear')
        degree = data.get('degree', 3)
        sigma = data.get('sigma', 0.1)
        iteration_steps = data.get('iteration_steps', 100)

        X = np.array(data['X'])
        y = np.array(data['y'])

        model, y = train_model(model_type, X, y, param, kernel, degree, sigma, iteration_steps, weights, metric)
        if model is None:
            return jsonify({'error': 'Failed to train model'}), 500

        plot_url = create_plot(model, X, y, model_type)
        return jsonify({'plot_url': plot_url})
    except Exception as e:
        logging.error(f"Error in model_csv API endpoint: {e}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

def create_csv_plot(X, y):
    try:
        fig, ax = plt.subplots()
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, edgecolors='k')
        legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
        ax.add_artist(legend1)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title('Uploaded CSV Data')

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close(fig)
        return plot_url
    except Exception as e:
        logging.error(f"Error creating plot for CSV data: {e}")
        return None
    
def generate_random_data(model_type, randomize_type):
    random_seed = random.randint(0, 10000)  # Use a random seed for each generation
    np.random.seed(random_seed)
    try:
        if randomize_type == 'normal':
            if model_type == 'kmeans':
                X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=random_seed)
            else:
                X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=random_seed)
        elif randomize_type == 'uniform':
            X = np.random.uniform(-10, 10, size=(100, 2))
            y = np.random.randint(0, 2 if model_type != 'kmeans' else 3, size=100)
        elif randomize_type == 'gaussian':
            X = np.random.normal(0, 1, size=(100, 2))
            y = np.random.randint(0, 2 if model_type != 'kmeans' else 3, size=100)
        elif randomize_type == 'exponential':
            X = np.random.exponential(1, size=(100, 2))
            y = np.random.randint(0, 2 if model_type != 'kmeans' else 3, size=100)
        else:
            raise ValueError("Unknown randomize type")
        
        logging.debug(f"Generated data with seed {random_seed}: X={X}, y={y}")
        return {'X': X.tolist(), 'y': y.tolist()}
    except Exception as e:
        logging.error(f"Error generating random data for {model_type}: {e}")
        return {'X': [], 'y': []}

def train_model(model_type, X, y, param, kernel, degree, sigma, iteration_steps, weights, metric):
    try:
        if model_type == 'knn':
            model = KNeighborsClassifier(n_neighbors=int(param), weights=weights, metric=metric)
        elif model_type == 'svm':
            if kernel == 'poly':
                model = SVC(C=float(param), kernel=kernel, degree=int(degree))
            elif kernel == 'rbf':
                model = SVC(C=float(param), kernel=kernel, gamma=float(sigma))
            else:
                model = SVC(C=float(param), kernel=kernel)
        elif model_type == 'kmeans':
            model = KMeans(n_clusters=int(param), max_iter=int(iteration_steps), n_init=1, init='random', random_state=42)
            y = model.fit_predict(X)
        elif model_type == 'dtree':
            model = DecisionTreeClassifier(max_depth=int(param))
        elif model_type == 'rforest':
            model = RandomForestClassifier(n_estimators=int(param), max_depth=int(degree))
        elif model_type == 'logreg':
            model = LogisticRegression(C=float(param), max_iter=int(iteration_steps))
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        model.fit(X, y)
        accuracy = model.score(X, y) * 100
        return model, y
    except Exception as e:
        logging.error(f"Error training {model_type} model: {e}")
        return None, None

def create_plot(model, X, y, model_type):
    try:
        fig, ax = plt.subplots()
        plot_decision_boundary(model, X, y, ax, model_type)
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close(fig)
        return plot_url
    except Exception as e:
        logging.error(f"Error creating plot for {model_type}: {e}")
        return None

def plot_decision_boundary(model, X, y, ax, model_type):
    try:
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

        if model_type == 'svm':
            Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, levels=[-1, 0, 1], alpha=0.3, colors=['#FF8080', '#8080FF', '#80FF80'])
            ax.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=['--', '-', '--'], colors='k')
            ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')
        elif model_type == 'kmeans':
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Spectral)
            
            # Plot the centroids with an icon
            centroids = model.cluster_centers_
            ax.scatter(centroids[:, 0], centroids[:, 1], s=200, c='yellow', marker='*', edgecolors='black', linewidths=2, zorder=5)
        else:
            if model_type in ['knn', 'dtree', 'rforest', 'logreg']:
                Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Spectral)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

        ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap=plt.cm.Spectral)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
    except Exception as e:
        logging.error(f"Error plotting decision boundary for {model_type}: {e}")

if __name__ == '__main__':
    app.run(debug=True)
