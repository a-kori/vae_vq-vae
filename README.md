# ✨ VAE & VQ-VAE
This repository contains the implementation code for two deep learning models: VAE (Variational Autoencoder) and VQ-VAE (Vector Quantized Variational Autoencoder), as well as two project presentations. [VAE & VQ-VAE Presentation](https://docs.google.com/presentation/d/1fI02-Wt4sg5LyUvZ6UsPsrUpli0r2j5q_zDvQM30nbY/edit?usp=sharing) explains the theory behind the models, and the [Project Summary](https://docs.google.com/presentation/d/1JfyOiWrc3Ve9UE3ykqmekwoqZ3dvaH12NE32zj0jLNU/edit?usp=sharing) describes the implementation process and the results of our work.

The project was implemented in the context of the Undergraduate Project "Machine Learning" at TU Dortmund and supervised by Marc Höftmann and Sebastian Konietzny. It is based on the papers:
- “Auto-Encoding Variational Bayes” by Diederik P. Kingma and MaxWelling,
- “Neural Discrete Representation Learning” by Aaron van den Oord, Oriol Vinyals and Koray Kavukcuoglu

## Running the code
To train the models, you can execute the `train.ipnyb` notebook found in `VAE` / `VQ-VAE` directories. Before running the code, make sure that the other Python files are included in the same directory, as they contain dependencies. You can also find the logs and the results of an example run on the CIFAR-10 dataset for each model in `train.ipnyb` and in the `output` directory.

Feel free to adjust the dataset and the number of epochs as you like. You can train the VAE on MNIST and CIFRAR-10 datasets, VQ-VAE can also be trained on CelebA. We recommend training the models on a GPU instead of a CPU for faster execution, e.g. on Google Colab.

## Authors
Anastasiia Korzhylova, Ivan Shishkin, Ramneek Agnihotri, Rodi Mehi
