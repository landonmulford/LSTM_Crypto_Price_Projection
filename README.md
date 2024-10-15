NOTE: Data used is from [https://www.kaggle.com/competitions/cryptocurrency-price-direction-predictor/data?select=train.csv](url) which is too large to upload to Github. Code will not run without it.

### Overview 
This model makes predictions about price and movement of a crypto token using a LSTM. The model takes in two inputs: the price of the token at a given time, and a variable called target. 
Target is given a value of 1 if the price moved up since the last time stamp, and 0 if it moved down or did not change. In my model, both of these variables were used for training, and the accuracy of the model
was based on the output of both of these variables. I tested the accuracy of the price by an average percent error. For the target, I set any prediction>0.5 to be a predicited 1, and <0.5 to be a predicted 0. Then I measured
the percentage of correct predictions.

### The Model
I used a LSTM, where for each data point, I set a lookback, which signified how many of the previous data points would be used to predict the given data point. While having a fixed length is not neccesary for LSTMs,
it made sense in my model. I also used a custom loss function, rather than a typical MSE error.

### Challenges
The main challenges in the model arised from the low signal to noise ratio present in the data. This caused predictions to be meaningless, with the model predicting the same result (up or down)
for every piece of data. To address this challenege, I used a custom loss function, where I both favored predictions that were near 0.5, and gave extra favor to those which got the "right" prediction.
This way, data would clump near the middle, which allowed predictions>0.5 (up), or <0.5 (down), but would also have strong preference to get the correct prediction.

### Results
Price error of ~1%, and correct target projection rate of ~55%
