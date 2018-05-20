# weight

Example SHAInet model to perform linear regression using Height/Weight.

Below is a Keras equivalent model:
```python
import pandas as pd
from keras.models import Sequential 
from keras.layers import Dense
from keras.optimizers import SGD

df = pd.read_csv('./data/weight-height.csv')

X = df[['Height']].values
Y = df['Weight'].values

model = Sequential()
model.add(Dense(1, input_shape=(1,)))
model.compile(SGD(lr=0.01), 'mean_squared_error')
model.fit(X, Y, epochs=40)
model.predict([75])
```

## Installation

Requires Crystal 0.24.2

## Usage

`crystal src/weight.cr`

## Development

Experimenting with different models.  Currently Adam is failing with NaN errors.  SGDM seems to provide fairly accurate results.

## Contributing

1. Fork it ( https://github.com/drujensen/weight/fork )
2. Create your feature branch (git checkout -b my-new-feature)
3. Commit your changes (git commit -am 'Add some feature')
4. Push to the branch (git push origin my-new-feature)
5. Create a new Pull Request

## Contributors

- [drujensen](https://github.com/drujensen) Dru Jensen - creator, maintainer
