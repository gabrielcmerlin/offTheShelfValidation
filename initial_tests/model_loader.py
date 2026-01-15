from models.fcn_biggerKernels import FCNRegressor_bk
from models.fcn_moreKernels import FCNRegressor_mk
from models.fcn_moreLayers import FCNRegressor_ml
from models.fcn import FCNRegressor

from models.fcn_biggerKernels_do import FCNRegressor_bk_do
from models.fcn_moreKernels_do import FCNRegressor_mk_do
from models.fcn_moreLayers_do import FCNRegressor_ml_do
from models.fcn_do import FCNRegressor_do

def load_models(model_name, input_channels):
    model = None

    if model_name == 'fcn_biggerKernels':
        model = FCNRegressor_bk(input_channels)
    elif model_name == 'fcn_moreKernels':
        model = FCNRegressor_mk(input_channels)
    elif model_name == 'fcn_moreLayers':
        model = FCNRegressor_ml(input_channels)
    elif model_name == 'fcn':
        model = FCNRegressor(input_channels)
    else:
        print('Model is not valid')

    return model