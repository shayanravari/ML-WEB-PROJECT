document.addEventListener('DOMContentLoaded', function () {
    const modelTypeSelect = document.getElementById('model-type');
    const knnControls = document.getElementById('knn-controls');
    const knnWeightsSelect = document.getElementById('knn-weights');
    const knnMetric = document.getElementById('knn-metric');
    const svmControls = document.getElementById('svm-controls');
    const svmDegreeControls = document.getElementById('svm-degree-controls');
    const svmSigmaControls = document.getElementById('svm-sigma-controls');
    const kmeansControls = document.getElementById('kmeans-controls');
    const dtreeControls = document.getElementById('dtree-controls');
    const rforestControls = document.getElementById('rforest-controls');
    const knnParamSlider = document.getElementById('knn-param');
    const knnParamValue = document.getElementById('knn-param-value');
    const svmParamSlider = document.getElementById('svm-param');
    const svmParamValue = document.getElementById('svm-param-value');
    const svmKernelSelect = document.getElementById('svm-kernel');
    const svmDegreeSlider = document.getElementById('svm-degree');
    const svmDegreeValue = document.getElementById('svm-degree-value');
    const svmSigmaSlider = document.getElementById('svm-sigma');
    const svmSigmaValue = document.getElementById('svm-sigma-value');
    const kmeansParamSlider = document.getElementById('kmeans-param');
    const kmeansParamValue = document.getElementById('kmeans-param-value');
    const kmeansIterStepSlider = document.getElementById('kmeans-iter-step');
    const kmeansIterStepValue = document.getElementById('kmeans-iter-step-value');
    const dtreeParamSlider = document.getElementById('dtree-param');
    const dtreeParamValue = document.getElementById('dtree-param-value');
    const rforestParamSlider = document.getElementById('rforest-param');
    const rforestParamValue = document.getElementById('rforest-param-value');
    const rforestDepthSlider = document.getElementById('rforest-depth');
    const rforestDepthValue = document.getElementById('rforest-depth-value');
    const logregControls = document.getElementById('logreg-controls');
    const logregParamSlider = document.getElementById('logreg-param');
    const logregParamValue = document.getElementById('logreg-param-value');
    const logregIterStepSlider = document.getElementById('logreg-iter-step');
    const logregIterStepValue = document.getElementById('logreg-iter-step-value');
    const ldaControls = document.getElementById('lda-controls');
    const ldaShrinkageSlider = document.getElementById('lda-shrinkage');
    const ldaShrinkageValue = document.getElementById('lda-shrinkage-value');
    const ldaSolverSelect = document.getElementById('lda-solver');
    const plotImg = document.getElementById('plot');
    const randomizeTypeSelect = document.getElementById('randomize-type');
    const randomizeButton = document.getElementById('randomize-button');
    const modelDescriptionContent = document.getElementById('model-description-content');
    const modelDescriptions = {
        knn: `<h3 class="text-md font-medium">K-Nearest Neighbors (KNN)</h3>
              <p class="text-sm text-gray-700 mb-4">The K-Nearest Neighbors (KNN) algorithm is a non-parametric, instance-based learning method used for classification and regression. The algorithm classifies data points based on the majority vote of their 'k' nearest neighbors in the feature space.</p>
              <h2 class="text-lg font-bold mb-4">Parameters</h2>
              <p class="text-sm text-gray-700 mb-4"><strong>k:</strong> The number of nearest neighbors to consider for making predictions.</p>
              <p class="text-sm text-gray-700 mb-4"><strong>Weights:</strong> The weight function used in prediction. Options include 'uniform' (all neighbors are weighted equally) and 'distance' (closer neighbors have more influence).</p>
              <p class="text-sm text-gray-700 mb-4"><strong>Metric:</strong> The distance metric used to calculate the distance between points. Common metrics include 'minkowski', 'euclidean', and 'manhattan'.</p>`,
        
        svm: `<h3 class="text-md font-medium">Support Vector Machine (SVM)</h3>
              <p class="text-sm text-gray-700 mb-4">Support Vector Machine (SVM) is a powerful supervised learning algorithm used for classification and regression tasks. It works by finding the hyperplane that best separates the data into different classes, maximizing the margin between them.</p>
              <h2 class="text-lg font-bold mb-4">Parameters</h2>
              <p class="text-sm text-gray-700 mb-4"><strong>C:</strong> The regularization parameter. It controls the trade-off between achieving a low error on the training data and minimizing the model complexity.</p>
              <p class="text-sm text-gray-700 mb-4"><strong>Kernel:</strong> The function used to transform the data into a higher-dimensional space. Common kernels include 'linear', 'polynomial', 'RBF', and 'sigmoid'.</p>
              <p class="text-sm text-gray-700 mb-4"><strong>Degree:</strong> The degree of the polynomial kernel function (if 'poly' is selected as the kernel).</p>
              <p class="text-sm text-gray-700 mb-4"><strong>Sigma (gamma):</strong> The kernel coefficient for 'RBF', 'poly', and 'sigmoid' kernels. It defines how far the influence of a single training example reaches.</p>`,
        
        kmeans: `<h3 class="text-md font-medium">K-Means Clustering</h3>
                 <p class="text-sm text-gray-700 mb-4">K-Means is an unsupervised learning algorithm used to partition data into 'k' clusters. Each data point is assigned to the cluster with the nearest centroid, and the centroids are recalculated iteratively until convergence.</p>
                 <h2 class="text-lg font-bold mb-4">Parameters</h2>
                 <p class="text-sm text-gray-700 mb-4"><strong>k:</strong> The number of clusters to form.</p>
                 <p class="text-sm text-gray-700 mb-4"><strong>Iteration Steps:</strong> The maximum number of iterations allowed for the algorithm to converge.</p>`,
        
        dtree: `<h3 class="text-md font-medium">Decision Tree</h3>
                <p class="text-sm text-gray-700 mb-4">A Decision Tree is a tree-like model used for classification and regression. It splits the data into subsets based on feature values, creating branches that represent the decision rules leading to different outcomes.</p>
                <h2 class="text-lg font-bold mb-4">Parameters</h2>
                <p class="text-sm text-gray-700 mb-4"><strong>Max Depth:</strong> The maximum depth of the tree. It controls the number of splits in the tree, preventing overfitting by limiting the tree's growth.</p>`,
        
        rforest: `<h3 class="text-md font-medium">Random Forest</h3>
                  <p class="text-sm text-gray-700 mb-4">Random Forest is an ensemble learning method that builds multiple decision trees during training and outputs the mode of the classes (classification) or mean prediction (regression) of the individual trees. It reduces overfitting and improves model accuracy.</p>
                  <h2 class="text-lg font-bold mb-4">Parameters</h2>
                  <p class="text-sm text-gray-700 mb-4"><strong>Number of Estimators (n):</strong> The number of trees in the forest.</p>
                  <p class="text-sm text-gray-700 mb-4"><strong>Max Depth:</strong> The maximum depth of the individual trees in the forest.</p>`,
        
        logreg: `<h3 class="text-md font-medium">Logistic Regression</h3>
                 <p class="text-sm text-gray-700 mb-4">Logistic Regression is a statistical model used for binary classification tasks. It models the probability that a given input point belongs to a particular class, using the logistic function to squeeze the output of a linear equation into a probability value.</p>
                 <h2 class="text-lg font-bold mb-4">Parameters</h2>
                 <p class="text-sm text-gray-700 mb-4"><strong>C:</strong> The inverse of regularization strength. Smaller values specify stronger regularization.</p>
                 <p class="text-sm text-gray-700 mb-4"><strong>Iteration Steps:</strong> The maximum number of iterations allowed for the optimization algorithm to converge.</p>`
    };
    
    let randomData = {};

    const THROTTLE_DURATION = 100;  // Global throttle duration in milliseconds

    function throttle(func, limit) {
        let lastFunc;
        let lastRan;
        return function(...args) {
            const context = this;
            if (!lastRan) {
                func.apply(context, args);
                lastRan = Date.now();
            } else {
                clearTimeout(lastFunc);
                lastFunc = setTimeout(function() {
                    if ((Date.now() - lastRan) >= limit) {
                        func.apply(context, args);
                        lastRan = Date.now();
                    }
                }, limit - (Date.now() - lastRan));
            }
        };
    }

    function updateModelDescription() {
        const selectedModel = modelTypeSelect.value;
        modelDescriptionContent.innerHTML = modelDescriptions[selectedModel];
    }

    function updateUI() {
        const modelType = modelTypeSelect.value;
        knnControls.style.display = modelType === 'knn' ? 'block' : 'none';
        svmControls.style.display = modelType === 'svm' ? 'block' : 'none';
        kmeansControls.style.display = modelType === 'kmeans' ? 'block' : 'none';
        dtreeControls.style.display = modelType === 'dtree' ? 'block' : 'none';
        rforestControls.style.display = modelType === 'rforest' ? 'block' : 'none';
        logregControls.style.display = modelType === 'logreg' ? 'block' : 'none';

        const svmKernel = svmKernelSelect.value;
        svmDegreeControls.style.display = (modelType === 'svm' && svmKernel === 'poly') ? 'block' : 'none';
        svmSigmaControls.style.display = (modelType === 'svm' && svmKernel === 'rbf') ? 'block' : 'none';
    }

    async function updateModel() {
        const modelType = modelTypeSelect.value;
        let param;
        let kernel;
        let degree;
        let sigma;
        let iteration_steps = 1;
        let shrinkage = null;
        let solver = 'svd';
        if (modelType === 'knn') {
            param = knnParamSlider.value;
            weights = knnWeightsSelect.value;
            metric = knnMetric.value;
        } else if (modelType === 'svm') {
            param = svmParamSlider.value;
            kernel = svmKernelSelect.value;
            degree = (kernel === 'poly') ? svmDegreeSlider.value : null;
            sigma = (kernel === 'rbf') ? svmSigmaSlider.value : null;
        } else if (modelType === 'kmeans') {
            param = kmeansParamSlider.value;
            iteration_steps = kmeansIterStepSlider.value;
        } else if (modelType === 'dtree') {
            param = dtreeParamSlider.value;
        } else if (modelType === 'rforest') {
            param = rforestParamSlider.value;
            degree = rforestDepthSlider.value;
        } else if (modelType === 'logreg') {
            param = logregParamSlider.value;
            iteration_steps = logregIterStepSlider.value;
        } else if (modelType === 'lda') {
            shrinkage = ldaShrinkageSlider.value;
            solver = ldaSolverSelect.value;
            if (solver === 'svd') shrinkage = null;
        }

        const response = await fetch('/api/model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ model_type: modelType, param: param, weights: weights, metric: metric, kernel: kernel, degree: degree, sigma: sigma, iteration_steps: iteration_steps, shrinkage: shrinkage, solver: solver, randomData: randomData })
        });

        if (response.ok) {
            const data = await response.json();
            plotImg.src = 'data:image/png;base64,' + data.plot_url;
        } else {
            console.error('Failed to fetch the model plot');
        }
    }

    async function randomizeData() {
        const modelType = modelTypeSelect.value;
        const randomizeType = randomizeTypeSelect.value;  // Get the selected randomize type
        console.log(`Randomizing data for model type: ${modelType} with randomize type: ${randomizeType}`);
        
        const response = await fetch('/api/randomize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ model_type: modelType, randomize_type: randomizeType })  // Pass randomize type
        });

        if (response.ok) {
            randomData = await response.json();
            console.log('Random data received:', randomData);
            updateModel();
        } else {
            console.error('Failed to randomize data');
        }
    }

    modelTypeSelect.addEventListener('change', function () {
        updateUI();
        randomizeData();
    });

    knnParamSlider.addEventListener('input', function () {
        knnParamValue.textContent = knnParamSlider.value;
    });
    knnParamSlider.addEventListener('input', throttle(updateModel, THROTTLE_DURATION));

    svmParamSlider.addEventListener('input', function () {
        svmParamValue.textContent = svmParamSlider.value;
    });
    svmParamSlider.addEventListener('input', throttle(updateModel, THROTTLE_DURATION));

    svmKernelSelect.addEventListener('change', function () {
        updateUI();
        updateModel();
    });

    knnWeightsSelect.addEventListener('change', throttle(updateModel, THROTTLE_DURATION));

    knnMetric.addEventListener('change', throttle(updateModel, THROTTLE_DURATION));

    svmDegreeSlider.addEventListener('input', function () {
        svmDegreeValue.textContent = svmDegreeSlider.value;
    });
    svmDegreeSlider.addEventListener('input', throttle(updateModel, THROTTLE_DURATION));

    svmSigmaSlider.addEventListener('input', function () {
        svmSigmaValue.textContent = svmSigmaSlider.value;
    });
    svmSigmaSlider.addEventListener('input', throttle(updateModel, THROTTLE_DURATION));

    kmeansParamSlider.addEventListener('input', function () {
        kmeansParamValue.textContent = kmeansParamSlider.value;
    });
    kmeansParamSlider.addEventListener('input', throttle(updateModel, THROTTLE_DURATION));

    kmeansIterStepSlider.addEventListener('input', function () {
        kmeansIterStepValue.textContent = kmeansIterStepSlider.value;
    });
    kmeansIterStepSlider.addEventListener('input', throttle(updateModel, THROTTLE_DURATION));

    dtreeParamSlider.addEventListener('input', function () {
        dtreeParamValue.textContent = dtreeParamSlider.value;
    });
    dtreeParamSlider.addEventListener('input', throttle(updateModel, THROTTLE_DURATION));

    rforestParamSlider.addEventListener('input', function () {
        rforestParamValue.textContent = rforestParamSlider.value;
    });
    rforestParamSlider.addEventListener('input', throttle(updateModel, THROTTLE_DURATION));

    rforestDepthSlider.addEventListener('input', function () {
        rforestDepthValue.textContent = rforestDepthSlider.value;
    });
    rforestDepthSlider.addEventListener('input', throttle(updateModel, THROTTLE_DURATION));
    
    logregParamSlider.addEventListener('input', function () {
        logregParamValue.textContent = logregParamSlider.value;
    });
    logregParamSlider.addEventListener('input', throttle(updateModel, THROTTLE_DURATION));

    logregIterStepSlider.addEventListener('input', function () {
        logregIterStepValue.textContent = logregIterStepSlider.value;
    });
    logregIterStepSlider.addEventListener('input', throttle(updateModel, THROTTLE_DURATION));

    randomizeButton.addEventListener('click', function () {
        randomizeData();
    });
    modelTypeSelect.addEventListener('change', updateModelDescription);

    updateUI();
    randomizeData();
    updateModelDescription();
});
