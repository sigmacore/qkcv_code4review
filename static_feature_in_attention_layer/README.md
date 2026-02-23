# static_feature_in_attention_layer
   
## Quick start:  
1. pre-install:
poetry==1.5.1

2. run notebook scripts  
  
## Introduction:  
The QKCV-modified attention based models are packaged in neuralforecast-1.7.5-qkcv.tar. Use it to overwrite your already installed neuralforecast-1.7.5:

```pip install --force-reinstall --no-deps  neuralforecast-1.7.5-qkcv.tar```

A new parameter v_qkcv has been added to new model classes:

```v_qkcv=1```, for version 1 in paper  
```v_qkcv=2```, for version 2 in paper  
```v_qkcv=3```, for version 3 in paper  
```v_qkcv=```any other int, for vanilla models  
  
Example:
```
from neuralforecast.models import TFT_v1

NeuralForecast(
    models=[TFT_v1(h=30, input_size=60,
                v_qkcv=1,
				...
                ),
    ],
    freq='D'
)

```
  
More examples in the .ipynb files.