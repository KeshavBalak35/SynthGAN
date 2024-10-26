# SynthGAN

## Overview

DiabetesGAN is a machine learning project that utilizes a Generative Adversarial Network (GAN) to create synthetic glucose level data for individuals with diabetes. This project aims to augment the PIMA Indians Diabetes Dataset with realistic, generated data to improve diabetes research and predictive modeling.

## Features

- Implements a GAN architecture for generating synthetic glucose level time series
- Utilizes the PIMA Indians Diabetes Dataset as a foundation
- Produces personalized glucose profiles based on individual characteristics
- Generates realistic glucose level fluctuations and patterns

## Dataset

The project uses the PIMA Indians Diabetes Dataset, which includes several health-related variables for individuals and their diabetes diagnosis status. This dataset is the basis for training the GAN and validating the synthetic data generation process.

## Model Architecture

The GAN consists of two main components:

1. **Generator**: Creates synthetic glucose level time series
2. **Discriminator**: Distinguishes between real and generated glucose data

Both networks are trained adversarially to improve the quality and realism of the generated data

## License

This project is licensed under the [MIT License](link-to-license-file).
