from tasker._base_tasker import _Tasker_base
from model.Homo import HomographyNet
class Train_Homo_and_save(_Tasker_base):
    def __init__(self, args):
        '''
        self.args
        self.experiment_dir
        self.save_path
        self.device
        self.logger
        '''
        pass
        super().__init__(args)
        print('Initialization complete.')
    
    def run(self):
        print('run!')
        self.model = HomographyNet(False)