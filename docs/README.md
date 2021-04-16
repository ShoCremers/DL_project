# Day Ahead Pricing Forecasts for Short Term Scheduling in Power Markets - A Deep Learning based approach

## For Brightspace Submission (not for blog)

Akshit Gupta (5137012) - a.gupta-20@student.tudelft.nl

Sho Cremers (5052602) - s.a.cremers@student.tudelft.nl

### Overview

Overall, the workload was divided equally among the two of us. While Sho worked mainly on preprocessing, Akshit worked on buiding the BLSTM model. The workload of training, evaluation, and blog writing was divided equally. 

## Start of blog

Written by Akshit Gupta and Sho Cremers

In this blog, we will go over deep learning-based RNNs (specifically LSTMs) to forecast day-ahead electricity prices in the context of power markets. The work is mainly based on the approach highlighted in [1] and uses publicly available real-world datasets for weather and electricity prices for training and evaluation. Our results show that RNNs (bidirectional LSTMs) are a powerful tool for forecasting electrical prices with quantifiable uncertainties. In doing so, we were successfully able to replicate the results of [1], albeit on a different dataset. All our code is open sourced and available on Github. All our code is open-sourced and available on Github.

All our code is open source and available on Github [here](https://github.com/ShoCremers/DL_project).

## Motivation (The who, what and why?)

In the modern world, electricity is a tradable quantity. The central authority regulates power grids in most countries to ensure transparency in the power markets. Since electricity storage is still costly, a balance between supply (generation at source) and demand (consumption at the sink) is desired in power grids. Due to these constraints, electricity generators and consumers (hereon, called energy actors as per [1]) submit pricing bids to the central authority at a fixed particular time of the day based on their multi-step ahead probabilistic forecasts to maximise their return on investment. These electricity prices are dependent on a large number of extrinsic factors such as renewable energy generation, weather conditions, and the season of the year.

For those interested, an in-depth explanation of these concepts is given [here](https://www.cmegroup.com/education/courses/introduction-to-power/understanding-basics-of-the-power-market0.html).
| ![Prediction Horizon](./images/predictionHorizon.png?raw=true) | 
|:--:| 
| *Prediction Horizon of Interest [1] * |

So, this work caters to the energy actors ("who") as described in the preceding paragraph who want accurate electricity pricing ("what) in power grids a day in advance from a model in order to maximise their RoI ("why").

## Theory ( The How? P1)
Being almost a trillion-dollar industry [2], introduced only a few decades ago, naturally, few million smart people in the world have come up with various ways to get the most accurate pricing forecasts. Some of the earlier and current works in this domain, such as ARMA (Autoregressive Moving Average), ARIMA (Auto Regressive Integrated Moving Average), Markov chains, etc., rely heavily on mathematical modelling but do not consider the uncertainty associated with various extrinsic factors. However, our focus here is to consider this problem first as a black box and then apply the most intelligible tool (Deep Learning-based RNNs) to solve it. In the context of RNNs, LSTM models are designed to select and propagate the most relevant contextual information automatically and are more flexible than concrete mathematical models of predefined complexity.

#### RNNs and LSTMs
In the field of Deep Learning (DL), Recurrent neural networks are well known to learn from time-series data when the dataset is large enough. However, traditional RNNs are known to suffer from vanishing and exploding gradients, preventing them from modelling time dependencies that are more than a few steps long. Further, RNNs process the inputs in sequential order and ignore the information from time steps in the future. In order to tackle these two problems, bidirectional Long Short Term Memory (BLSTMs, a type of RNNs) are used to remember the sequential pattern of interest over these arbitrary long time intervals along with support for exploiting information over the whole temporal horizon. Thus, making them an apt choice for the problem at hand. Bi-directional RNNs can be considered as having two separate layers. The forward layer takes inputs in the given order. The backward layer takes the inputs in the reversed order. Then, both of these layers are connected to the output layer. Speech recognition tasks have found it beneficial to include both directions in their models since words can be better recognised using the whole sentence rather than just previous words. Hence, our intuition is that, in addition to past weather and day-ahead prices giving an indication of future day ahead electricity prices, future weather and day-ahead prices can also help to model the past day-ahead prices. The image below is the visualisation of a bidirectional RNN.

| ![BRNN](./images/BRNN.png?raw=true) | 
|:--:| 
| Bidirectional RNN [1]   |


Refer to Andrew Ng's renowned videos [3] for an in-depth explanation of these concepts.

#### Quantiles and loss function
While we want the most accurate pricing forecast, the uncertainty in these forecasts or predictions can provide valuable information to improve decision making and bidding. And uncertainty is quantified mathematically using probability. From the domain of probability, we use the non-parametric model of prediction errors as defined in [1]. We use quantiles as a way to quantise the uncertainty in our predictions using quantile regression. By definition, a quantile is " a location within a set of ranked numbers, below which a certain proportion, p, of that set lie." Thus, instead of just one concrete pricing value as output, there will be multiple quantiles as the outputs of our model. The model is trained to minimise the quantile loss (or pinball loss) in order to output the most optimal quantiles. The total loss is the sum over all specified quantiles of interest.


| ![Quantile Loss](./images/lossFuntion.png?raw=true) | 
|:--:| 
| *Quantile Loss Function [1]  * |

Refer to [this](https://www.youtube.com/watch?v=IFKQLDmRK0Y) video by StatQuest to know more about quantiles.


## Implementation (The How? P2)

#### Dataset and Preprocessing
We used the hourly data obtained from [5] for Belgium's past historical electrical prices for the last six years, i.e., from Jan 5, 2015 - Dec 31, 2020. Belgium was chosen as the paper[1] also uses the same country for evaluation. This data contained 6 null values when daylight savings time starts for every year. To factor for the null values, second-degree polynomial interpolation was used. On the other hand, there was once in a year in which 2 values existed for the same hour when the daylight savings time ended. To account for this, only the maximum of the 2 values was kept.

`day_ahead['Day-ahead Price [EUR/MWh]'] = day_ahead['Day-ahead Price [EUR/MWh]'].interpolate(method='polynomial', order=2)`
`day_ahead = day_ahead.groupby('MTU (CET)')['Day-ahead Price [EUR/MWh]'].max().reset_index()`

In addition to the electrical pricing data, the hourly weather data was also obtained for Brussels from [6] for the same time period, and both the weather and pricing datasets were combined to form the input dataset. The features that were used from the weather contain temperature, wind speed, degree of the wind direction precipitation, humidity, and air pressure.

`total = pd.merge(day_ahead, weather, how='outer', on='datetime')`

Since the predictions have to take place at 12 pm each day, a sliding window approach with a configurable sequence length was used. Hence, the training samples (X, Y) to the network have the below form (assuming sequence length of 24):
X: [Weather, Price etc. at t=12:00 (the previous day), Weather, Price etc. at t=13:00, Weather, Price etc. at t=14:00,..... Weather, Price etc. at t=11:00 (the current day)]
Y: [Price at t=0:00, Price at Price at t=1:00,.......Price at t=11:00 (the next day)] 

After preprocessing the dataset, the data from Jan 2015 - Oct 2020 was used for training, the data for Nov 2020 was used for validation, and data for Dec 2020 was used for testing.

#### Feature Engineering

Time (hourly), day of the week, week, and current month are cyclic features, so they need to be represented appropriately so that the model is aware of their cyclic nature. Similarly to [1], three different representations were used for time. One of them was incremental indexing within the range of [0.1,2.4]. Another representation of time was 5 inputs of binary gray coded time in the range of [(0,0,0,0,1), (1,0,1,0,0)]. Finally, we used the mutually exclusive binary representation with 24 inputs (one-hot encoding of time). During the experiment, datasets with each of the time variables were tested, as well as the dataset with no time variable. The dataset incremental indexing time representation did better on the validation set, and hence no time variable was used on the evaluation, which will be explained later on.

To represent the day of the week, week of the month, and current month, they were converted to two periodic waveforms of sine and cosine function with appropriate periods. This is shown in the code block below, and a good explanation of this kind of representation for these cyclic features is given in [9]. 
```
    df1['dow_sin'] = np.sin(df1.index.dayofweek*(2.*np.pi/7))
    df1['dow_cos'] = np.cos(df1.index.dayofweek*(2.*np.pi/7))

    df1['mnth_sin'] = np.sin((df1.index.month-1)*(2.*np.pi/12))
    df1['mnth_cos'] = np.cos((df1.index.month-1)*(2.*np.pi/12))

    df1['week_sin'] = np.sin((pd.Int64Index(df1.index.isocalendar().week)-1)*(2.*np.pi/53))
    df1['week_cos'] = np.cos((pd.Int64Index(df1.index.isocalendar().week)-1)*(2.*np.pi/53))
```

To reduce the risk of features having different scales, which could cause prioritizing certain features falsly, features were either normalized or standardized. Variables like degree of wind direction and humidity, which have clear minimum and maximum values (wind degree: 0 - 360, humidity: 0 - 100), were normalized. Precipitation was also normalized by taking the minimum and maximum values from the training data but applying them to the whole data. For the remaining variables that were not binary, standardization was applied, which transforms the features to have the mean of 0 and the standard deviation of 1. 


#### Criterion and Optimiser

As described earlier, we are interested in quantiles for output and hence, use the QuantileLoss function provided by Pytorch Forecasting [7]. We had also tested custom implementation of quantile loss by referencing [8], and it gave similar results. Adam was used as the optimiser, and the learning rate was kept at 0.01 during the experiment with other parameters of Adam at their standard values.

#### The Model Architecture
A bidirectional LSTM model is used with the number of outputs equal to the number of quantiles of interest. The model contains a variable number of LSTM layers (hyperparameter) followed by a linear layer for each quantile output. The code is mostly straightforward, with the model expecting the number of hidden layers (num_layers), the dimensionality of each hidden layer, the output dimensionality, and an array of quantiles in the constructor. The output dimensionality is kept 24 in our case due to prediction for next 24 hours and the quantiles of interest are chosen to be [0.01,0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99] in line with the paper[1].
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

 #### Training the model
The model was trained for 500 epochs with a batch size of 64. The following shows the training and validation loop for the final hyperparameters that were obtained. Here patient_cnt is used for early stopping which is explained in the upcoming section.

```
for t in range(num_epochs): 
                    
    err = []
                    
    # training
    for batch in trainloader:
        inputs, outputs = batch
        model.add_noise_to_weights() # adding noise to lstm weights during the training
        y_train_pred = model(inputs)

        loss = torch.mean(torch.sum(criterion.loss(torch.transpose(y_train_pred,1,2), outputs), dim=2))
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        err.append(loss.item())

    # validation
    with torch.no_grad():
        reds=model(x_val)
        val_loss = torch.mean(torch.sum(criterion.loss(torch.transpose(preds,1,2), y_val), dim=2)).item()
        val_losses.append(val_loss)

    if val_loss < val_loss_best:
        val_loss_best = val_loss
        patience_cnt = 0
    else:
        patience_cnt +=1
        if patience_cnt == patience:
            print("Early stopping: Epoch ", t+1, "training loss: ", sum(err)/len(err), "validation loss: ", val_loss)
            break
                    
    if (t+1) % 10 == 0:
        print("Epoch ", t+1, "training loss: ", sum(err)/len(err), "validation loss: ", val_loss)
```

 #### Regularisation
Two regularisation techniques were used during the training. One was the addition of the noise on the LSTM weights and biases. Gaussian noise with mean of 0 and standard deviation of 0.01 was added to make the model more robust from the noise in the data. The following shows the implementation of weight noise in the model. 

```
def add_noise_to_weights(self):
    with torch.no_grad():
        # add noise to lstm weights
        for weights in model.lstm._all_weights:
            for weight in weights:
                noise = torch.normal(0, 0.01, size=self.lstm._parameters[weight].size())
                self.lstm._parameters[weight].add_(noise)
```

Another regularisation we used was early stopping. Training stopped when the validation set's quantile loss did not decrease in the last 10 epochs since the lowest validation loss. 

#### (Hyper)parameter Tuning
To determine the best parameter and hyperparameter setting, we used a nested loop as shown in the figure below. The parameters that had to be determined were the time variable, the sequence length of previous hours, the size of the hidden dimension, and the number of layers. The time variables that were tested were, no time variable, incremental indexing, gray code binary, and mutually exclusive binary. We tested with the previous 12, 24, 36, 48, and 72 hours for the sequence length. The tested hidden dimension sizes were 4, 8, 16, 32, 64, and 128. Finally, we tested the model with 1-4 layers. As per [1] and theory, since our training dataset is not huge, our initial intuition is that the dimensionality of hidden layers should be kept small in order to avoid overfitting.

We used the quantile loss on the validation set to evaluate the best parameter setting. As mentioned earlier, we used the patience size of 10 for early stopping, and so the performance of the model was determined by the average validation loss of the last 10 epochs. This was to avoid choosing a model that did well just one time, but instead, a model that consistently did well.

| ![loop](./images/tuningLoop.png?raw=true) | 
|:--:| 
| Hyperparameter Tuning loop (inspired from [1])   |

The best model was chosen to have incremental indexing, the sequence length of 72 hours, 32 hidden dimensions, and 3 layers. Having obtained these parameters, the model was trained again and was evaluated on the testing set. 

#### Post Processing
Once the quantiles of the day ahead electricity price of the testing set has been predicted using the trained model, the quantiles were inverse-transformed, so the values are in terms of day-ahead price in euros. 

## Results
| ![PaperResults](./images/paperResults.png?raw=true) | 
|:--:| 
| *Results from Paper [1]* |

| ![ourResults](./images/result_full.png?raw=true) | 
|:--:| 
| *Our Results* |

| ![ourResults](./images/result_7days.png?raw=true) | 
|:--:| 
| *Our Results: First 7 days* |

Our results show that the model can predict the price quite well in general. Looking at the first seven days, the prediction during the day seems to show that the actual values are mostly within the quantiles. However, it has difficulty in predicting prices of early morning. It is likely that early morning electricity prices did not have a large spread in the training data, hence predicting a smaller region with high confidence. 

| Model         | Quantile loss |
| ------------- |:-------------:|
| Paper [1]     | 28.00 &euro;  |
| Ours          | 28.17 &euro;  | 

Comparing the average quantile loss between [1] and our model, they perform almost equally. However, it is difficult to get the full picture since [1] only shows the plot of 7 days and not for the whole month tested. Our results show that predicting the price during the holidays can be more difficult, so it would have been nice if we could compare the performance.

## Ambiguities

While we aimed to reproduce [1] as best as we can, we encountered several ambiguities. The paper did not specify how and what kind of noises were added to the weights for the regularisation. Besides mentioning that they used early stopping, they also did not mention how it was executed. They also do not clearly state how the data were divided into training, validation, and testing sets. They used the whole "month of winter 2017" for evaluation but decided only to show a plot of prediction of seven days. This can be an issue since we can expect that results will be worse during the week with Christmas break. Finally, a large number of explanatory variables were used in their experiment, such as Solar PV generation, wind generation, public holidays, etc., but their encoding in input was unspecified. Finally, no concrete hyperparameters have been specified in the paper. 

## Final Words
Following the approach of [1], the probabilistic forecast of electricity prices was reproduced on a different dataset with a limited number of explanatory variables. Even with these constraints, Our resultant output curves are similar to [1], while using a much simpler model. 

To improve the model, we can incorporate additional features, such as future weather forecasts. Another addition that can be beneficial would be a variable indicating whether the day is a holiday, as our prediction was worse during the holidays. However, adding more of these features may also make the model suffer from the curse of dimensionality, so their addition should be complemented with additional training data.

Complete source code is available [here](https://github.com/ShoCremers/DL_project).

## References
1. J. Toubeau, J. Bottieau, F. Vallée and Z. De Grève, "Deep Learning-Based Multivariate Probabilistic Forecasting for Short-Term Scheduling in Power Markets" in IEEE Transactions on Power Systems, vol. 34, no. 2, pp. 1203–1215, March 2019, doi: 10.1109/TPWRS.2018.2870041.
2. https://www.alliedmarketresearch.com/renewable-energy-market
3. https://www.coursera.org/lecture/nlp-sequence-models/recurrent-neural-network-model-ftkzt
4. https://www.youtube.com/watch?v=IFKQLDmRK0Y
5. https://transparency.entsoe.eu/transmission-domain/r2/dayAheadPrices/show
6. https://www.worldweatheronline.com/brussels-weather-history/be.aspx
7. https://pytorch-forecasting.readthedocs.io/en/latest/api/pytorch_forecasting.metrics.QuantileLoss.html
8. https://github.com/maxmarketit/Auto-PyTorch/blob/develop/examples/quantiles/Quantiles.ipynb
9. http://blog.davidkaleko.com/feature-engineering-cyclical-features.html



