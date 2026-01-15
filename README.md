# Off-the-shelf Algorithms for Classification and Extrinsic Regression

## Introduction

Author: Gabriel da Costa Merlin - ICMC/USP

Advisor: Diego Furtado Silva - ICMC/USP

Project financed by Research Support Foundation of the State of São Paulo (FAPESP).

Summary: This repository contains code and experiments from a FAPESP Undergraduate Research project on the off-the-shelf use of deep learning models for time series, focusing on Time Series Classification (TSC) and Time Series Extrinsic Regression (TSER). The Fully Convolutional Network (FCN) is used as a reference architecture and evaluated in its standard form and with simple structural variations. Results show that while the base FCN is competitive, small architectural changes can improve performance, especially in TSER. Additional experiments explore replacing the prediction head with Kolmogorov–Arnold Networks (KAN), including hybrid and end-to-end setups. Experiments were conducted at scale using public datasets (128 TSC and 19 TSER), with results analyzed using cross-validation and statistical tests. The KAN part of this project was published in the proceedings of ENIAC 2025.

## Folders

- initial_tests: exploratory TSC/TSER experiments and analyses used to validate the project hypotheses on small-scale setups
- tser_comparison: larger-scale and more robust TSER experiments designed to further evaluate the hypotheses

**Note:** KAN-related code is available at https://github.com/gabrielcmerlin/FCKAN
