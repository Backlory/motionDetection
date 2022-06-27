


from utils.conf import get_conf
import datetime
from algorithm.infer_all import Inference_all
from algorithm.infer_Homo_cancel import Inference_Homo_cancel
from algorithm.infer_Region_Proposal_cancel import Inference_Region_Proposal_cancel
 
if __name__ == "__main__":
    #path = r"E:\dataset\dataset-fg-det\all_x264.mp4"
    path = r"E:\dataset\dataset-fg-det\Janus_UAV_Dataset\train_video\video_all.mp4"
    args = get_conf('Inference_all')
    Tasker = Inference_all(args=args)

    
    #Tasker.infer_align = Inference_Homo_cancel()
    #Tasker.infer_RP = Inference_Region_Proposal_cancel()
     
    Tasker.run_test(dataset = path)
