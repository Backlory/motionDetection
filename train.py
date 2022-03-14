from tasker.train_Homo import Train_Homo_and_save
from utils.conf import get_conf


if __name__ == "__main__":
    #
    arg = get_conf('Train_Homo_and_save_1')
    Tasker = Train_Homo_and_save(arg)
    #
    Tasker.run()