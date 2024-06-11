## VAE Implementation: commit journal
...

08.06.2024 - Anastasiia: 
- implemented VAE training
- modified dataset visualization
- added training optimizations: optimizer selection, LR schedule, AMP scaler
- debugging
- fixed sampling function error

10.06.2024 - Anastasiia:
- adjusted code structure and reconstruction plotting to changes in VAE model
- further training improvements & bug fixes

11.06.2024 - Anastasiia:
- repo cleanup

08.06.2024 - Ramneek 
- implemented datasets.py
  
09.06.2024 - Ramneek
- added batch normalization and dropout to networks.py
- adjusted parameters, change scheduler , and added gradient clipping to training
  
10.06.2024 - Ramneek
-remove color jitter, change normalization in datasets.py

08.06.2024 - Rodi
- implemented the neural network (first and second) version
10.06.2024
- implemented test_notebook, as test environement to optimize our VAE Model
- added more convelutional layers to the network
- added dropout(p) to avoid overfitting
- added transform.resize() to use the same nn for CIFAR10 and MNIST
