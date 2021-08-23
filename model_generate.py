from model import *

# 输入的每一个小段的测量点数=Tokens数量，tokens数量固定，根据tokens数量决定要划分多少个小段
# 小段数量 = 测量点总数/固定的tokens数量， 根据病害分布确定tokens数量，预先定义为M
# 每多少个点，做一次检测
class ModelGenerator():
    def __init__(self, row_per_detect, num_event, num_tokens):
        super(ModelGenerator, self).__init__()
        self.num_tokens = num_tokens
        self.in_channel = row_per_detect
        self.num_class = num_event
    
    def DetectModel_lite(self, dropout=0.2, pre_train=False, state_dir=None):
        args = {
            'in_channel' : self.in_channel, 'd_model' : 256, 'heads' : 8, 'dim_head' : 64, 'dropout' : dropout, 
            'num_tokens' : self.num_tokens, 'exp_ratio' : 4, 'num_blocks' : 2, 'class_exp' : 2, 'num_class' : self.num_class
        }
        model = DetecModel(**args)
        if pre_train:
            model.load_state_dict(torch.load(state_dir))
            print('Model loaded.')
        return model

    def DetectModel_mid(self, dropout=0.3, pre_train=False, state_dir=None):
        args = {
            'in_channel' : self.in_channel, 'd_model' : 512, 'heads' : 8, 'dim_head' : 64, 'dropout' : dropout, 
            'num_tokens' : self.num_tokens, 'exp_ratio' : 4, 'num_blocks' : 4, 'class_exp' : 2, 'num_class' : self.num_class
        }
        model = DetecModel(**args)
        if pre_train:
            model.load_state_dict(torch.load(state_dir))
            print('Model loaded.')
        return model
    
    def DetectModel_large(self, dropout=0.4, pre_train=False, state_dir=None):
        args = {
            'in_channel' : self.in_channel, 'd_model' : 512, 'heads' : 8, 'dim_head' : 64, 'dropout' : dropout, 
            'num_tokens' : self.num_tokens, 'exp_ratio' : 4, 'num_blocks' : 6, 'class_exp' : 2, 'num_class' : self.num_class
        }
        model = DetecModel(**args)
        if pre_train:
            model.load_state_dict(torch.load(state_dir))
            print('Model loaded.')
        return model
    
    def DetectModel_custom(self, args, pre_train=False, state_dir=None):
        model = DetecModel(**args)
        if pre_train:
            model.load_state_dict(torch.load(state_dir))
            print('Model loaded.')
        return model


if __name__ == '__main__':
    print()
    print('########################## Inference Test ##########################')
    print()
    model = ModelGenerator(row_per_detect=700, num_event=12, num_tokens=90).DetectModel_large()
    input_tensor = torch.randn(32, 90, 700)
    scores = model(input_tensor)
    print('Input Size: ', input_tensor.shape)
    print('Output Size: ', scores.shape)
    print()
    print('##########################                ##########################')
    print()
