# Image Classification using AWS SageMaker

This project demonstrates the usage of Amazon SageMaker and other AWS services to build, train, deploy and monitor an image classification model that detects whether an image contains a dog or cat.

## Project Set Up and Installation
Explanation of the different files:
- train_model.py - a python module that has the model creation, training and validation functionalities.
- train_and_deploy.ipynb - Contains different scripts used for downloading and preprocessing the dataset, hyperparameter tuning, PyTorch Estimator creation and deployment purposes.

## Dataset
The dataset used in this project is downloaded from the famous [dogs-vs-cats competition](https://www.kaggle.com/c/dogs-vs-cats). Since the original dataset doesn't categorize the images into their respective categories based on the image content, I have written a script so that there would be two directories (one for dog and another one for cat images). Once that is done, the whole dataset is then split up into train and validation sets.

### Access
AWS CLI is used to upload the arranged data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data, when training the model. 

## Model

The model chosen in this project is a pretrained version of resnet50 as it is one of the most commonly used convolutional neural networks. It is trained on more than a million images from the [ImageNet](https://www.image-net.org/) database.

## Hyperparameter Tuning

The hyperparameters that are optimized include batch-size, learning-rate and the number of epochs the training has to run through. The objective metric is the average train loss and the objive is to find a Train Job that minimizes it.

Here is the screenshot of the hyperparameters tuning job:
![19145999-00c0938296f789a62f724188a8f53e19](https://user-images.githubusercontent.com/41271840/147415378-9578efeb-d6c7-45fd-8e66-2245b781c203.png)

The best training job and hyperparameters of it are queried like this.
#Get the best estimators and the best HPs
best_estimator = tuner.best_estimator()

#Get the hyperparameters of the best trained model
best_estimator.hyperparameters()

This gave the following output:

2021-12-25 08:31:18 Starting - Preparing the instances for training  
2021-12-25 08:31:18 Downloading - Downloading input data  
2021-12-25 08:31:18 Training - Training image download completed. Training in progress.  
2021-12-25 08:31:18 Uploading - Uploading generated training model  
2021-12-25 08:31:18 Completed - Training job completed  
{'_tuning_objective_metric': '"average test loss"',  
 'batch-size': '"256"',  
 'epochs': '4',  
 'lr': '0.00390315210654023',  
 'sagemaker_container_log_level': '20',  
 'sagemaker_estimator_class_name': '"PyTorch"',  
 'sagemaker_estimator_module': '"sagemaker.pytorch.estimator"',  
 'sagemaker_job_name': '"pytorch-training-2021-12-25-08-03-55-769"',  
 'sagemaker_program': '"train_model_.py"',  
 'sagemaker_region': '"us-east-1"',  
 'sagemaker_submit_directory': '"s3://sagemaker-us-east-1-807116804612/pytorch-training-2021-12-25-08-03-55-769/source/sourcedir.tar.gz"'}  

The best hyperparameters are: {'epochs': 4, 'batch-size': 256, 'lr': 0.00390315210654023}  
 
## Debugger Profiling
Model debugger profiling in sagemaker is achieved by specifying the profile and hook parameters in the lines defining the PyTorch estimator. The respective hooks for the training and testing phases are registered and used in the train_model.py script.

By enabling the SageMaker Debugger and Profiler, different debugging information logged while the training job was running and the resource consumption like model parameters, CPU and GPU usage etc are revealed in the output.

## Model Deployment
The deployed model can be queried by creating a Predictor object by specifying the endpointname as a parameter and the calling the `predict()` method on it providing a data (sample image). The image has to be preprocessed and transformed by the `create_data_loader()` function to be consumed by the predictor.

Screenshot of the endpoint in service.
![image](https://user-images.githubusercontent.com/41271840/147610383-2aa0e16a-c5ca-4d4d-ba13-f1785569780c.png)

## Model Performance
The model has an accuracy of 98% on the validation set, which is pretty much good.
