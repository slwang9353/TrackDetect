from data_format import format_trans_Numpy_norm_toTensor
from model_generate import *
import numpy as np


def inference(row_per_detect, num_event, len_subsegment, test=True):
    USE_GPU = True
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    model = ModelGenerator(
        row_per_detect=row_per_detect, num_event=num_event, num_tokens=len_subsegment
    ).DetectModel_large(
        dropout=0.4, pre_train=True, state_dir='./check_point/check_point.pth'
    ).to(device=device).eval()

    n = 1
    while n==1:
        if test:
            raw_data = np.random.randn(row_per_detect, 900)
    ################################################################################################## 
    ####                                                                                          ####
    ####       接上游数据，每stack row_per_detect行数据，以 numpy 格式赋给 raw_data                  ####
    ####                                                                                          ####
    ##################################################################################################

        input_tensor = format_trans_Numpy_norm_toTensor(
            raw_data, len_subsegment
        ).to(device=device)

        scores = model(input_tensor)
        _, preds = scores.max(1)
        preds = preds.numpy()
        if test:
            print()
            print('############################### Distribution of Events on the Track ###############################')
            print()
            print(preds)
            print()
            print(' ##################################################################################################')

if __name__ == '__main__':

    inference(row_per_detect=700, num_event=12, len_subsegment=90, test=True)

    

