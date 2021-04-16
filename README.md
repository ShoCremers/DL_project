# Day Ahead Pricing Forecasts for Short Term Scheduling in Power Markets — A Deep Learning based approach 

## To setup environment and install dependencies
conda env create -f env_config.yaml

## Project Structure 
- /data contains the dataset used
- /results contain the final results obtained 
- /BLSTM.py (depreciated) contains raw python code used for GCP N1 Machine to take advantage of CUDA
- /preprocessing.ipynb is the jupyter notebook for combining datasets and normalising features
- /BSLTM_experiment.ipynb is the jupyter notebook with all the code after preprocessing the dataset

All experiments run on M1 13 inch Macbook Pro (2020). The average time for training was 10 minutes.
The associated blog is available [here](https://akshitgupta1695.medium.com/day-ahead-pricing-forecasts-for-short-term-scheduling-in-power-markets-a-deep-learning-based-ddc6ca64a2cf)