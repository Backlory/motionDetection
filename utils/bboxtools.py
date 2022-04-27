import cv2

def ten2bboxs(ten):
    assert{False}
    assert(ten.shape[1] == 2)   #[n, 2, h, w]
    bboxes = ten.detach().cpu().numpy()
    return bboxes

def vis_bbox(img, bboxes):
    h_img,w_img,_ = img.shape
    for bbox in bboxes:
        x,y,w,h = bbox
        w = w * w_img
        h = h * h_img
        x = x * w_img - w/2
        y = y * h_img - h/2
        
        img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
    cv2.imshow("1", img)
    cv2.waitKey(0)
    return img