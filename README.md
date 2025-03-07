Place LIBWO.py, model.py, and ModelTrain.py in the same folder. After correctly downloading and configuring all datasets, ensure that the runtime environment is properly set up. Modify the dataset paths in the ModelTrain.py file. Then, run the ModelTrain.py file to initiate the entire training process.

First, invoke the LIBWO algorithm for hyperparameter tuning. Through multiple iterations, this algorithm intelligently searches for the global optimal solution, continuously optimizing the hyperparameters within the search space, and ultimately outputs the best hyperparameter combination and its corresponding fitness value.

Once LIBWO completes the optimization and generates the best hyperparameters, the system will automatically pass these parameters to the ResNet18 model. At this point, the model begins training, and the training process will continue until the model reaches convergence, ensuring optimal performance.

Upon completion of the training, the system will conduct model testing, outputting key metrics such as test accuracy, test loss, and additional metrics including precision, recall, and F1-score. These metrics validate the performance of the ResNet18 model on the target task. This comprehensive workflow ensures efficient model training and evaluation, achieving optimal classification performance.
