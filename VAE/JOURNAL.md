## VAE Implementation: contribution journal

08.06.2024 - Ramneek:
- implemented datasets.py

08.06.2024 - Rodi:
- implemented the neural network (first and second) version
- debugging

08.06.2024 - Ivan:
- implemented evaluate function
- implemented sampling function
- first logging implementation
- first sampling plot implementation
- time measurement and output

08.06.2024 - Anastasiia: 
- implemented VAE training
- modified dataset visualization
- added training optimizations: optimizer selection, LR schedule, AMP scaler
- debugging
- fixed sampling function error

09.06.2024 - Ramneek:
- added batch normalization and dropout to networks.py
- adjusted parameters, changed scheduler
- added gradient clipping to training

10.06.2024 - Ramneek:
- removed color jitter
- changed normalization in datasets.py

10.06.2024 - Rodi:
- implemented test_notebook as test environement to optimize our VAE Model
- added more convelutional layers to the network
- added dropout(p) to avoid overfitting
- added transform.resize() to use the same nn for CIFAR10 and MNIST

10.06.2024 - Anastasiia:
- adjusted code structure and reconstruction plotting to changes in VAE model
- further training improvements & bug fixes

11.06.2024 - Ramneek:
- adjusted parameters and loss as requested
- removed dropout
- added bias to linear layer

11.06.2024 - Ivan:
- nice-looking console printing
- frequent (every 150th batch) console printing
- 2D test loss and train loss plot
- 2D learning rate plot
- (Group-specific) 2D plot for each loss component
- return modification in loss_function

11.06.2024 - Anastasiia:
- repo cleanup
- testing
- fixed plotting errors