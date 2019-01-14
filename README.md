# Machine Learning Researcher 

This project is used to enable research and rapid prototyping of various machine learning algorithms.

We have decided to use TensorFlow, since we want to minimize the effort of taking the resulting models to production.

## Machine Learning Guideline

This guideline is intended to remind you of the necessities for going live with your model. 

+ Data flow
  + Data ingestion
  + Data validation
  + Data transformation (Feature engineering)
+ Model prototyping
  + Building model prototypes quickly
  + Interface for training and prediction data inputs
  + Usable for local testing but also easily to be migrated to production
+ Model analysis (evaluation and validation)
  + Visualizations (learning rate, error rate, success rate, prediction accuracy...)
  + Simulation of the prod environment
  + A/B testing
+ Model serving
  + Deploying new versions of your model
+ Model maintenance
  + Model drift handling 
    + Detecting changes in the data distribution
  + Continuous training 
    + Transfer learning (warm-starting)
    + Online training
  + Retraining
+ Monitoring
  + Error/Success rate
  + Drift detection and alerts
  + Business metrics (absolute and relative uplift, moving averages...)
  + Data flow
  + Instance healthiness

## Projects

### 1. New floor price optimisation algorithm based on neural networks

We use a decision tree algorithm for this problem, but since the performance has been decreasing we started this research project in the scope of my bachelor thesis

#### How to run

Run TensorFlow with Jupyter on Docker:
<pre class="prettyprint">
  <code class="devsite-terminal">docker run -it --rm --name ml-researcher -v PATH/TO/Mobfox-Stack/projects/ml-researcher/data/1_FloorPriceOprimisation:/notebooks -p 8888:8888 -p 6006:6006 tensorflow/tensorflow</code>
</pre>
Run TensorBoard for visualisations:
<pre class="prettyprint">
  <code class="devsite-terminal">docker exec -it ml-researcher tensorboard --logdir /notebooks/data/1_FloorPriceOprimisation/tmp/[REPLACE WITH DSP ID]/logs</code>
</pre>

Run following notebooks in the order provided:
1. "Preprocessing with TensorFlow Transform.ipynb"
2. "Custom DNN Regression Floor Price Optimisation.ipynb"
3. "Model Analysis.ipynb"

Author: Reshad Dernjani

### 2. Render rate optimisation

Future work!