# NLP Project: Sentiment Analysis with BERT and PSO

## Table of Contents
1. [Introduction](#introduction)
   - [Problem Statement](#problem-statement)
   - [What is PSO?](#what-is-pso)
   - [How to Use PSO?](#how-to-use-pso)
2. [Methodology](#methodology)
   - [The Fine-Tuned BERT Benchmark Model (LiYuan)](#the-fine-tuned-bert-benchmark-model-liyuan)
   - [The BERT Model With A New Layer Training](#the-bert-model-with-a-new-layer-training)
   - [The BERT Model with Two New Layers Training](#the-bert-model-with-two-new-layers-training)
   - [PSO-Based Hyperparameter Tuning](#pso-based-hyperparameter-tuning)
3. [Results](#results)
   - [The Test Results of the BERT Benchmark Model (LiYuan)](#the-test-results-of-the-bert-benchmark-model-liyuan)
   - [The Results of the PSO-Based Hyperparameter Tuning Model](#the-results-of-the-pso-based-hyperparameter-tuning-model)
   - [Comparing the Benchmark and New Model](#comparing-the-benchmark-and-new-model)
4. [Conclusions](#conclusions)
5. [References](#references)

---

## Introduction
In this project, we aimed to gain practical experience in developing a Natural Language Processing (NLP) model using Transformers. We utilized a fine-tuned version of BERT, a pre-trained model specifically tailored for Amazon reviews, with the objective of extending its applicability to perform sentiment analysis on review data from various websites. Additionally, we employed Particle Swarm Optimization (PSO), a widely used optimization technique, to search for optimal hyperparameters for our model. We adopted multiple strategies to address the challenges of hyperparameter optimization within the constraints of time and hardware resources.

### Problem Statement
We utilized the 'LiYuan/amazon-review-sentiment-analysis' model, a fine-tuned model trained specifically on Amazon review data. Our objectives were to evaluate its performance when applied to review data from different websites and to enhance its performance by optimizing its hyperparameters. We aimed to achieve this by training and testing the model with review data from diverse sources. Additionally, we provided two sets of optimal hyperparameters that were trained and compared with the performance of the original model.

### What is PSO?
Particle Swarm Optimization (PSO) is a population-based stochastic optimization technique inspired by the social behavior of bird flocking or fish schooling. The algorithm maintains a population of candidate solutions, referred to as particles, which move through the search space to find the optimal solution. Each particle's position in the search space represents a potential solution, and its movement is guided by its own experience and the collective behavior of the swarm. PSO does not require gradient information, making it well-suited for optimization problems where derivatives are not readily available or are expensive to compute.

### How to Use PSO?
In this project, we used the "GlobalBestPSO" method for implementing the PSO method. The "GlobalBestPSO" method is a variant of the Particle Swarm Optimization (PSO) algorithm that focuses on a global exploration of the search space to find the optimal solution. This method is particularly effective for problems where a global optimum is sought, as it encourages the entire swarm to converge towards the same solution.

---

## Methodology
We present three main models for training and evaluating results based on the LiYuan model:

### The Fine-Tuned BERT Benchmark Model (LiYuan)
We utilized a BERT model with a pre-trained 5-class SoftMax layer (LiYuan) to establish a benchmark performance on different databases. The LiYuan model is a fine-tuned version of the nlptown/bert-base-multilingual-uncased-sentiment model specifically tailored for sentiment analysis on Amazon US Customer Reviews.

### The BERT Model With A New Layer Training
For this model, we used a BERT model (LiYuan) with an un-trained 5-class SoftMax layer. The objective was to train this new layer using reviews from various websites and compare its performance with the benchmark model.

### The BERT Model with Two New Layers Training
In this model, we employed a BERT model (LiYuan) with two un-trained layers. The aim was to train these new layers using reviews from diverse websites, compare the results with the benchmark model, and analyze the impact of increased layer depth on the final results.

### PSO-Based Hyperparameter Tuning
This model utilized a BERT model (LiYuan) with an un-trained 5-class softmax layer, acting as a cost function for a PSO method. The goal was to search for optimal values of hyperparameters such as Learning Rate, Batch Size, and Weight Decay Ratio using the PSO algorithm.

---

## Results
We present the results of our validation and testing efforts across various datasets using predefined models.

### The Test Results of the BERT Benchmark Model (LiYuan)
We obtained the BERT benchmark model from the "Hugging Face" website and employed its classifier to assess our diverse datasets. The outcomes indicated that the benchmark model's accuracy in analyzing Amazon reviews falls below the 80% mark. Moreover, the accuracy and F1 scores for non-Amazon datasets notably lag behind those of Amazon, suggesting a struggle to effectively classify reviews from other platforms.

### The Results of the PSO-Based Hyperparameter Tuning Model
We employed the PSO method to identify the optimal hyperparameters for our model. We pursued two distinct strategies and obtained the most favorable results for each strategy. Subsequent adjustments to the resulting hyperparameters led us to select the following sets for fine-tuning the models. The results demonstrated that each model's performance varied across datasets, highlighting the model's greater sensitivity to epoch count and data size than to specific parameters.

### Comparing the Benchmark and New Model
When comparing the test results of the benchmark model with the fine-tuned model featuring one SoftMax layer, notable improvements were observed. These results demonstrate the success of the fine-tuning process and its potential applicability for reviewing data from various websites.

---

## Conclusions
Our project demonstrates the potential of leveraging pre-trained models like BERT in conjunction with optimization techniques such as PSO to enhance their performance in real-world applications. However, it also underscores the importance of understanding the limitations and nuances of such models when applied beyond their original scope. Further research could focus on refining the PSO-based hyperparameter tuning process and exploring additional strategies to improve the model's generalization capabilities across diverse datasets and domains.

---

## References
[1] LiYuan, Hugging Face, https://huggingface.co/LiYuan/amazon-review-sentiment-analysis/tree/main
[2] LoganKells, Hugging Face, https://huggingface.co/datasets/LoganKells/amazon_product_reviews_video_games
[3] Eswar Chand, Kaggle, https://www.kaggle.com/datasets/eswarchandt/amazon-music-reviews?resource=download
[4] Deniz Bilgin, Kaggle, https://www.kaggle.com/datasets/denizbilginn/google-maps-
