# Day Ahead Pricing Forecasts for Short Term Scheduling in Power Markets- A Deep Learning based approach

Written by Akshit Gupta and @shocremers

In this blog, we will go over deep learning based RNNs (specifically LSTMs) to forecast day ahead electricity prices in the context of power markets. The work is based largely on the approach highlighted in [1] and uses publicly available real world datasets for weather and electricity prices for training and evaluation. Our results show that RNNs ( bi-directional LSTMs) are a powerful tool for forecasting electrical prices with quantifable uncertainities. In doing so, we were successfully able to replicate the results of [1] albeit on a different dataset.

## Motivation (The who, what and why?)

In the modern world, electricity is a tradable quantity. To ensure transparency in the power markets, the power grids in most countries are regulated by a central authority. Since, storage of electricity is still costly, a balance between supply (generation at source) and demand (consumption at sink) is desired in power grids. Due to these constraints, electricity generators and consumers (hereon, called as energy actors as per [1]) submit pricing bids to the central authority at a fixed particular time of the day based on their multi step ahead probabilistic forecasts in order to maximise their return on investment. These electricity prices are dependent on a large number of extrinsic factors such as renewable energy generation, the weather conditions and the season of the year.

For those interested, an in-depth explanation of these concepts is given here.
| ![Prediction Horizon](./images/predictionHorizon.png?raw=true) | 
|:--:| 
| *Prediction Horizon of Interest [1] * |
So, this work caters to the energy actors ("who") as described in the preceding paragraph who want accurate electricity pricing ("what) in power grids a day in advance from a model in order to maximise their RoI ("why").

## Theory ( The How? P1)
Being almost a trillion dollar industry [2] introduced only a few decades ago, naturally, few million smart people in the world have come up with various ways to get the most accurate pricing forecasts. Some of the earlier and current works in this domain such as autoregressive moving average (ARMA), ARIMA, markov chains etc. rely heavily on mathematical modelling but do not take into account the uncertainty associated with various extrinsic factors. However, as lazy computer scientists (may remove this part from sentence), our focus here is to consider this problem first as a black box and then apply the most intelligible tool (Deep Learning based RNNs) to solve it. In the context of RNNs, LSTM models are utilised which are designed to automatically select and propagate the most relevant contextual information and are more flexible than concrete mathematical models of predefined complexity.

#### RNNs and LSTMs
In the field of Deep Learning (DL), Recurrent neural networks are well known to learn from time series data when the dataset is large enough. However, traditional RNNs are known to suffer from the problem of vanishing and exploding gradients preventing them to model time depencies which are more than a few steps long. Further, RNNs process the inputs in sequential order and ignore the information from time steps in the future. In order to tackle these two problems, bi-directional Long Short Term Memory (BLSTMs, a type of RNNs) are used to remember sequential pattern of interest over these arbitrary long time intervals along with support for exploiting information over the whole temporal steps. Thus, making them an apt choice for the problem at hand.
TODO (maybe): shall we explain BLSTM theory here? {lstm block, equation, bi directional network}
Refer to Andrew Ng's renowned videos [3] for in-depth explanation of these concepts.
#### Quantiles and loss function
While we want the most accurate pricing forecast, the uncertainty in these forecasts or predictions can provide valuable information to improve decision making and bidding. And uncertainty is quantified mathematically using probability. From the domain of probability, we use the non parametric model of predictions errors as defined in [1], and use quantiles as a way to quantise the uncertainty in our predictions using quantile regression. By definition, a quantile is " a location within a set of ranked numbers, below which a certain proportion, p, of that set lie". Thus, instead of just one concrete pricing value as output, there will be multiple quantiles as the outputs of our model. The model is trained to minimise the quantile loss (or pinball loss) in order to output the most optimal quantiles. The total loss is the sum over all specified qauntiles of interest.


| ![Quantile Loss](./images/lossFuntion.png?raw=true) | 
|:--:| 
| *Qauntile Loss Function [1]  * |

Refer to this video by StatQuest to know more about quantiles.


## Implementation (The How? P2)

#### Dataset and Preprocessing
[5],[6]
For the past historical electrical prices, we used the hourly data obtained from [5] for the last 6 years i.e. from Jan 5 2015 - Dec 31 2020 for Belgium. Belgium was chosen as [1] also uses the same country for evaluation. This data contained 6 null values when daylight savings time start for every year. To factor for the null values, polynomial interpolation was used with degree of 2.
Similary, the points in year when daylight savings time ended had 2 values. To account for this, only the maximum of the 2 values was kept.

`day_ahead['Day-ahead Price [EUR/MWh]'] = day_ahead['Day-ahead Price [EUR/MWh]'].interpolate(method='polynomial', order=2)`
`day_ahead = day_ahead.groupby('MTU (CET)')['Day-ahead Price [EUR/MWh]'].max().reset_index()`

In additon to the electrical pricing data, the hourly weather data was also obtained for Brussels from [6] for the same time period and both the weather and pricing datasets were combined to form the input dataset.

`total = pd.merge(day_ahead, weather, how='outer', on='datetime')`

In contrast to [1], the exact hour for each day was encoded with incremental indexing within a continous range of [0.1,2.4]. We also tried the mutually exclusive binary representation as done in [1] and found the the performance detoriated slightly for the same test set. Hence, incremental indexing representation of daily hours was chosen in the end.
```
time = (total['time'].values/100).astype(int)
time_increment = time/10`
```

-Add info about Normalisation here

Since the predictions have to take place at 12pm each day, a sliding window approach with a configurable sequence length (36 in below code snippet) was used. Hence, the training samples (X,Y) to the network have the below form:
X: [Weather and Price at t=12:00, Weather and Price at t=11:00, Weather and Price at t=10:00,..... Weather and Price at t=00:00 (the previous day)]
Y: [Price at t=13:00, Price at Price at t=14:00,.......Price at t=12:00 (the next day)] 
After preprocessing the dataset, similar to [1], the data from Jan 2015 - Oct 2020 was used for training, the data for Nov 2020 was used for validation and data for Dec 2020 was used for testing.

#### Criterion and Optimiser

As described earlier, we are interested in quantiles for output and hence, use the inbuilt QuantileLoss function provided by Pytorch. We had also tested custom implementation of quantile loss by referencing [7] and it gave similar results. Adam was used as the optimiser and the learning rate was kept 0.01

#### The Model Architecture
A bidirection LSTM model is used with the number of outputs equal to the number of quantiles of interest. The model contains a variable number of LSTM layers (hypermparameter) followed by a linear layer for each quantile output. The code is mostly straightforward with the model expecting the number of hidden layers (num_layers), the dimensionality of each hidden layer, the output dimensionality and an array of quantiles in the constructor. The output dimensionality is kept 24 in our case due to prediction for next 24 hours and the qauntiles of interest are chosen to be [.01,0.05, 0.10,0.25, .5, 0.75, 0.90, 0.95, .99] in line with the paper[1].
```
class BLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, quantiles):
        super(BLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        self.out_shape = len(quantiles)
        final_layers = [ nn.Linear(hidden_dim*2, output_dim) for _ in range(len(self.quantiles))]
        self.final_layers = nn.ModuleList(final_layers)
        
   def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_dim).requires_grad_() #hidden layer output
        # Initialize cell state
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_dim).requires_grad_() 
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # Index hidden state of last time step        
        return torch.stack([layer(out[:, -1, :]) for layer in self.final_layers], dim=1)
 ```
 As per [1] and theory, since our training dataset is not huge, the dimensionality of hidden layers should be kept small in order to avoid overfitting.
 
#### Post Processing
#### Regularisation
early stopping and dropout
#### Hyperparameter Tuning
settings used


#### Results
| ![PaperResults](./images/paperResults.png?raw=true) | 
|:--:| 
| *Results from Paper [1]* |

| ![ourResults](./images/e%3D10%2Ch%3D3%2Chd%3D20%2Cinput%3DpriceAndWeather.png?raw=true) | 
|:--:| 
| *Our Results* |

## Ambiguities

- Sequence Length is unspecified
- The tuning parameters for early stopping and gaussian noise unspecified
- How the dataset was broken into train, test and validation
- A large number of explanatory variables used, such as Solar PV generation, wind generation, public holidays etc. but their encoding in input unspecified.
- The final image does not contain the week of prediction in the month of December. It is expected that results will be worse during the week with christmas break.

## Final Words
Following the approach of [1], the probabilistic forecast of electricity prices was reproduced on a different dataset with limited number of explanatory variables. Even with these constraints, our resultant output curves are similar to [1], thus highlighting the generalisability of the approach of the paper.

## References
1. J. Toubeau, J. Bottieau, F. Vallée and Z. De Grève, "Deep Learning-Based Multivariate Probabilistic Forecasting for Short-Term Scheduling in 	   Power Markets" in IEEE Transactions on Power Systems, vol. 34, no. 2, pp. 1203–1215, March 2019, doi: 10.1109/TPWRS.2018.2870041.
2. https://www.alliedmarketresearch.com/renewable-energy-market
3. https://www.coursera.org/lecture/nlp-sequence-models/recurrent-neural-network-model-ftkzt
4. https://www.youtube.com/watch?v=IFKQLDmRK0Y
5. https://transparency.entsoe.eu/transmission-domain/r2/dayAheadPrices/show
6. https://www.worldweatheronline.com/brussels-weather-history/be.aspx
7. https://github.com/maxmarketit/Auto-PyTorch/blob/develop/examples/quantiles/Quantiles.ipynb



