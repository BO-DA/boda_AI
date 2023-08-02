import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import ImageDraw, Image
import sys
sys.path.append("..")
from MobileSAM.mobile_sam import sam_model_registry, SamPredictor
from VanishingPoint.main import GetLines,GetVanishingPoint
import math

# SAM
sam_checkpoint = "MobileSAM/weights/mobile_sam.pt"
model_type = "vit_t"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

def show_mask2(mask):
    color = np.array([30, 144, 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    return mask_image[:,:,:3]

def fline(x1, y1, x2, y2):
    # 두 점을 지나는 직선의 기울기를 구합니다.
    if x2 - x1 == 0:
        raise ValueError('오류')
    m = (y2 - y1) / (x2 - x1)
    # y절편을 구합니다. (한 점을 대입하여 b를 구할 수 있습니다.)
    b = y1 - m * x1
    return m, b
    
def cross(m1,b1,m2,b2):
    x = (b2-b1)/(m1-m2)
    y = m1*x+b1
    return x, y
    
def bisecting_line(m1, m2, b1, b2, x, y):
    # 두 직선의 각을 구합니다
    angle1 = math.atan2(1,m1)
    angle2 = math.atan2(1,m2)
            
    # 두 각의 평균을 구해 이등분하는 각의 크기를 계산합니다
    bisecting_angle = (angle1 + angle2)/2

    # 이등분하는 각의 크기를 이용하여 이등분하는 직선의 기울기를 구합니다
    bisecting_slope = math.tan(bisecting_angle)

    # 이등분하는 직선의 y절편을 계산합니다
    bisecting_y_intercept = y - bisecting_slope * x

    # 이등분하는 직선의 방정식 문자열 생성

    return bisecting_slope, bisecting_y_intercept


class Safe_Zone():

    def __init__(self,image):
        self.image = image
    
    def SAM(self):
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        sam.to(device=device)
        sam.eval()
        predictor = SamPredictor(sam)

        predictor.set_image(image)

        input_point = np.array([[image.shape[1]/2-400, image.shape[0]-10],[image.shape[1]/2-200, image.shape[0]-200], 
                        [image.shape[1]/2, image.shape[0]-10], [image.shape[1]/2+200, image.shape[0]-200],
                        [image.shape[1]/2+400, image.shape[0]-10]])
        input_label = np.array([1, 1, 1, 1, 1])

        masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,)

        mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask

        masks2, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        mask_input=mask_input[None, :, :],
        multimask_output=False,)

        save_mask = show_mask2(masks[2]).astype(np.uint8)

        return save_mask, masks2

    def VanishingPoint(self,save_mask):
        Lines = GetLines(save_mask)
        VanishingPoint = GetVanishingPoint(Lines)
        return VanishingPoint
    
    def Angular_Bisector(self,masks2,VanishingPoint):

        mask_x = np.sum(masks2,axis = 1)[0]
        for i in range(len(mask_x[::-1])):
            if mask_x[::-1][i] !=0:
                max_x = self.image.shape[1]-1-i
                break
        for i in range(len(mask_x)):
            if mask_x[i] !=0:
                min_x = i
                break
        for i in range(len(masks2[0,:,max_x])):
            if masks2[0,:,max_x][i] != False:
                max_y = i
                break
        for i in range(len(masks2[0,:,min_x])):
            if masks2[0,:,min_x][i] != False:
                min_y = i
                break
        
        image = Image.fromarray(self.image)

        mid_x, mid_y =  int(VanishingPoint[0]), int(VanishingPoint[1])

        m1,b1 = fline(mid_x,mid_y,min_x,min_y)
        m2,b2 = fline(mid_x,mid_y,max_x,max_y)

        mm, bm = bisecting_line(m1, m2, b1, b2,mid_x,mid_y)
        x_mid = -bm/mm;
        x1 = (np.array(image).shape[0]-b1)/m1
        x2 = (np.array(image).shape[0]-b2)/m2

        draw = ImageDraw.Draw(image)
        #draw.line((x_mid,1080,x,y), fill="green", width=5)
        draw.line(((x1+x_mid)/2,np.array(image).shape[0],mid_x, mid_y), fill="green", width=10)
        draw.line(((x2+x_mid)/2,np.array(image).shape[0],mid_x, mid_y), fill="green", width=10)

        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

