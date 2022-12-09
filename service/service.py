import json

import bentoml
import numpy as np
import pandas as pd
from bentoml.io import PandasDataFrame, Multipart, NumpyNdarray, JSON

from NewsSortingService import NewsSortingService

PYTORCH_MODEL_TAG = "st_pytorch_model:q2adbglw2g22mncb"
TOKENIZER_TAG = "st_tokenizer:rgsjxblw2g2t2ncb"

model = bentoml.pytorch.load_model(PYTORCH_MODEL_TAG)
tokenizer = bentoml.picklable_model.load_model(TOKENIZER_TAG)

news_sorting_service = bentoml.Service("news_sorting_service")

@news_sorting_service.api(input=JSON(), output=PandasDataFrame())
def predict(input_data: dict):


    df = pd.DataFrame.from_dict(input_data)

    ns_service = NewsSortingService()

    to_predict = df.head(10).text.apply(lambda x: x.replace("\n", " ")).tolist()
    pred, model_outputs = ns_service.predict(to_predict=to_predict, tokenizer=tokenizer, model=model)
    pred_df = pd.DataFrame(data=pred)
    # model_output_df = pd.DataFrame(data=model_outputs)
    # df = pd.concat([pred_df, model_output_df], axis=0)
    return pred_df
