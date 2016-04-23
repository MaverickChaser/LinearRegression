Kaggle - Job Salary Prediction
==============================

This repo contains sample code for the [Job Salary Prediction competition](https://www.kaggle.com/c/job-salary-prediction/), hosted by [Kaggle](http://www.kaggle.com) with [Adzuna](http://www.adzuna.co.uk/).

It contains several benchmarks marked as different git tags. This is the **Mean Benchmark**. For another example, see the [Random Forest Benchmark](https://github.com/benhamner/JobSalaryPrediction/tree/RandomForestBenchmark).

Executing this benchmark requires Python 2.7 along with the following packages:

 - pandas (version >=10.1)
 - numpy

To run the benchmark,

1. [Download the data](https://www.kaggle.com/c/job-salary-prediction/data)
2. Modify SETTINGS.json to point to the training and validation data on your system, as well as a place to save the trained model and a place to save the submission
3. Train the model by running `python train.py`
4. Make predictions on the validation set by running `python predict.py`
5. [Make a submission](https://www.kaggle.com/c/job-salary-prediction/team/select) with the output file

This benchmark took under a minute to execute on a Windows 8 laptop with 8GB of RAM and 4 cores at 2.7GHz.
