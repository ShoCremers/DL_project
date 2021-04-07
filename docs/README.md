# Day Ahead Pricing Forecasts for Short Term Scheduling in Power Markets- A Deep Learning based approach

Written by Akshit Gupta and @shocremers

In this blog, we will go over deep learning based RNNs (specifically LSTMs) to forecast day ahead electricity prices in the context of power markets. The work is based largely on the approach highlighted in [1] and uses publicly available real world datasets for weather and electricity prices for training and evaluation. Our results show that RNNs ( bi-directional LSTMs) are a powerful tool for forecasting electrical prices with quantifable uncertainities. In doing so, we were successfully able to replicate the results of [1] albeit on a different dataset.

## Motivation (The who, what and why?)

In the modern world, electricity is a tradable quantity. To ensure transparency in the power markets, the power grids in most countries are regulated by a central authority. Since, storage of electricity is still costly, a balance between supply (generation at source) and demand (consumption at sink) is desired in power grids. Due to these constraints, electricity generators and consumers (hereon, called as energy actors as per [1]) submit pricing bids to the central authority at a fixed particular time of the day based on their multi step ahead probabilistic forecasts in order to maximise their return on investment. These electricity prices are dependent on a large number of extrinsic factors such as renewable energy generation, the weather conditions and the season of the year.

For those interested, an in-depth explanation of these concepts is given here.
TODO: ADD IMAGE HERE
So, this work caters to the energy actors ("who") as described in the preceding paragraph who want accurate electricity pricing ("what) in power grids a day in advance from a model in order to maximise their RoI ("why").

## Theory ( The How? P1)
Being almost a trillion dollar industry [2] introduced only a few decades ago, naturally, few million smart people in the world have come up with various ways to get the most accurate pricing forecasts. Some of the earlier and current works in this domain such as autoregressive moving average (ARMA), ARIMA, markov chains etc. rely heavily on mathematical modelling but do not take into account the uncertainty associated with various extrinsic factors. However, as lazy computer scientists (may remove this part from sentence), our focus here is to consider this problem first as a black box and then apply the most intelligible tool (Deep Learning based RNNs) to solve it. In the context of RNNs, LSTM models are utilised which are designed to automatically select and propagate the most relevant contextual information and are more flexible than concrete mathematical models of predefined complexity.

#### RNNs and LSTMs
In the field of Deep Learning (DL), Recurrent neural networks are well known to learn from time series data when the dataset is large enough. However, traditional RNNs are known to suffer from the problem of vanishing and exploding gradients preventing them to model time depencies which are more than a few steps long. Further, RNNs process the inputs in sequential order and ignore the information from time steps in the future. In order to tackle these two problems, bi-directional Long Short Term Memory (BLSTMs, a type of RNNs) are used to remember sequential pattern of interest over these arbitrary long time intervals along with support for exploiting information over the whole temporal steps. Thus, making them an apt choice for the problem at hand.
TODO (maybe): shall we explain BLSTM theory here? {lstm block, equation, bi directional network}
Refer to Andrew Ng's renowned videos [3] for in-depth explanation of these concepts.
#### Quantiles and loss function
While we want the most accurate pricing forecast, the uncertainty in these forecasts or predictions can provide valuable information to improve decision making and bidding. And uncertainty is quantified mathematically using probability. From the domain of probability, we use the non parametric model of predictions errors as defined in [1], and use quantiles as a way to quantise the uncertainty in our predictions using quantile regression. By definition, a quantile is " a location within a set of ranked numbers, below which a certain proportion, p, of that set lie". Thus, instead of just one concrete pricing value as output, there will be multiple quantiles as the outputs of our model. The model is trained to minimise the quantile loss (or pinball loss) in order to output the most optimal quantiles. The total loss is the sum over all specified qauntiles of interest.
TODO: Add equation here
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



