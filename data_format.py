from einops import rearrange
import numpy as np
import torch
from model_generate import ModelGenerator

def format_trans_Numpy_norm_toTensor(raw_data, num_tokens):
    '''raw_data (N, D) -> (n m d)'''
    '''d = N, m = num_tokens, n = D / m'''
    raw_data = (raw_data - np.mean(raw_data, axis=0)) * (np.var(raw_data, axis=0) ** -0.5)
    raw_data = torch.Tensor(raw_data)
    return rearrange(raw_data, 'N ( n m ) -> n m N', m=num_tokens)


if __name__ == '__main__':
    print()
    print('########################## Inference Test ##########################')
    print()
    row_per_detect = 700
    num_measure = 9000
    num_tokens = 90
    num_event = 12
    print('Split to %d sub-segment' % (num_measure / num_tokens))
    raw_data = np.random.randn(row_per_detect, num_measure)
    input_tensor = format_trans_Numpy_norm_toTensor(raw_data, num_tokens)
    model = ModelGenerator(row_per_detect=row_per_detect, num_event=num_event, num_tokens=num_tokens).DetectModel_large()
    scores = model(input_tensor)
    torch.save(model.state_dict(), './check_point/check_point.pth')
    print('Raw Data Size: ', raw_data.shape)
    print('Input Size: ', input_tensor.shape)
    print('Output Size: ', scores.shape)
    print()
    print('##########################                ##########################')
    print()
