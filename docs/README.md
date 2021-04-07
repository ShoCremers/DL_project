# Day Ahead Pricing Forecasts for Short Term Scheduling in Power Markets- A Deep Learning based approach

Written by Akshit Gupta and @shocremers

In this blog, we will go over deep learning based RNNs (specifically LSTMs) to forecast day ahead electricity prices in the context of power markets.The work is based largely on the approach highlighted in [1] and uses publicly available real world datasets for electricity prices and weather for training and evaluation. Our results show that RNNs ( bi-directional LSTMs) are a powerful tool for forecasting and in turn, highlight the learning

## Motivation (The who, what and why?)

In the modern world, electricity is a tradable quantity. To ensure transparency in the power markets, the power grids in most countries is regulated by a central authority. Storage of electricity is still costly, hence, a balance between supply (generation at source) and demand (consumption at sink) is desired in power grids. Due to these constraints, electricity generators and consumers (hereon, called as participants) submit pricing bids to the central authority at a fixed particular time in the day based on their day ahead forecasting models to maximise there return on investment. For those interested, an in-depth explanation of these concepts is given here.
TODO: ADD IMAGE HERE
So, this work caters to the participants ("who") as described in the preceding paragraph who want accurate electricity pricing ("what) in power grids a day in advance from a model in order to maximise their RoI ("why").

## Theory ( The How? P1)
Being almost a trillion dollar industry [2] introduced only a few decades ago, naturally, few million smart people in the world have come up with various ways to get the most accurate pricing forecasts. Some of the earlier and current works in this domain rely heavily on stochastic theory, mathematics and a lot of mind (EWMA, ADD MORE METHODS HEER etc). However, as lazy computer scientists, our focus is to consider this problem first as a black box and then apply the most intelligible tool (Deep Learning based RNNs) to solve it.

#### RNNs and LSTMs
In the field of Deep Learning (DL), Recurrent neural networks are well known to learn from sequential data. However, traditional RNNs are known to suffer from the problem of vanishing and exploding gradients in case the sequential pattern of interest occurs with large temporal resolution (or time intervals or gap length) . In order to tackle this, Long Short Term Memory (LSTMs, a type of RNNs) are used to remember sequential pattern of interest over these arbitrary time intervals. Thus, making them an apt choice for the problem at hand.
Refer to Andrew Ng's renowned videos [3] for in-depth explanation of these concepts.
#### Quantiles
While we want the most accurate pricing forecast, the uncertainty in these forecasts or predictions can provide valuable information to improve decision making and bidding. And uncertainty is quantified mathematically using probability. From the domain of probability, as done in [1], we use quantiles as a way to quantise the uncertainty in our predictions. By definition, a quantile is " a location within a set of ranked numbers, below which a certain proportion, p, of that set lie". Thus, instead of just one concrete pricing value as outputs, there will be multiple quantiles as the output of our model.
Refer to this video by StatQuest to know more about quantiles.


## Implementation (The How? P2)

#### Dataset
[5],[6]
#### Preprocessing
#### Normalisation
#### Loss Function
explain quantile loss
#### The Model Architecture
explain model architecure and code
#### Post Processing
#### Regularisation
early stopping and dropout
#### Hyperparameter Tuning
#### Results
comparison to original paper

## Final Words

## References
1. J. Toubeau, J. Bottieau, F. Vallée and Z. De Grève, "Deep Learning-Based Multivariate Probabilistic Forecasting for Short-Term Scheduling in 	   Power Markets" in IEEE Transactions on Power Systems, vol. 34, no. 2, pp. 1203–1215, March 2019, doi: 10.1109/TPWRS.2018.2870041.
2. https://www.alliedmarketresearch.com/renewable-energy-market
3. https://www.coursera.org/lecture/nlp-sequence-models/recurrent-neural-network-model-ftkzt
4. https://www.youtube.com/watch?v=IFKQLDmRK0Y
5. https://transparency.entsoe.eu/transmission-domain/r2/dayAheadPrices/show
6. https://www.worldweatheronline.com/brussels-weather-history/be.aspx



