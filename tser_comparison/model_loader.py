from models.bn.fcn_biggerKernels import FCNRegressor_bk
from models.bn.fcn_moreKernels import FCNRegressor_mk
from models.bn.fcn_moreLayers import FCNRegressor_ml
from models.bn.fcn import FCNRegressor

from models.do.fcn_biggerKernels_do import FCNRegressor_bk_do
from models.do.fcn_moreKernels_do import FCNRegressor_mk_do
from models.do.fcn_moreLayers_do import FCNRegressor_ml_do
from models.do.fcn_do import FCNRegressor_do

def load_models(model_name, input_channels):
    model = None

    if model_name == 'FCN_biggerKernels':
        model = FCNRegressor_bk(input_channels)
    elif model_name == 'FCN_moreKernels':
        model = FCNRegressor_mk(input_channels)
    elif model_name == 'FCN_moreLayers':
        model = FCNRegressor_ml(input_channels)
    elif model_name == 'FCN':
        model = FCNRegressor(input_channels)
    else:
        print('Model is not valid')

    return model