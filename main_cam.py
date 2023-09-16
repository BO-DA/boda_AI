import cv2
import cv2
from SafeZone import Safe_Zone 
import argparse
import numpy as np

def extract_masked_region(image, mask):
    # Mask 값이 1인 픽셀만 추출하여 새로운 이미지 생성
    masked_image = np.copy(image)
    masked_image[mask != 1] = 0

    return masked_image

webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

while webcam.isOpened():
    status, frame = webcam.read()

    frame_index = 0

    if status:
        safe = Safe_Zone(frame)
        save_mask, masks2 = safe.SAM()

        vs = safe.VanishingPoint_Triangle(masks2)

        if vs is None or vs[0] == 0 or vs[1] == 0:
            try:
                vs = vs_previous
            except:
                vs = [int(frame.shape[0]/2),int(frame.shape[1]/2)]
        else:
            vs_previous = vs

        if frame_index == 0:
            frame, pr_mask, pr_x1, pr_x2 = safe.Angular_Bisector(masks2,vs)
        else:
            frame, pr_mask, pr_x1, pr_x2 = safe.Angular_Bisector(masks2,vs,pr_mask,pr_x1,pr_x2)
        
        masks_save = np.expand_dims(masks2, axis=2)
        masks_save = np.squeeze(masks_save)
        masked_region = extract_masked_region(frame, masks_save)

        
        if frame.shape[1]/2 < pr_x1:
            print('right')
            dirg = 'right'
        elif frame.shape[1]/2 > pr_x2:
            print('left')
            dirg = 'left'
        else:
            print('Normal')
            dirg = 'Normal'
        
        cv2.circle(masked_region, (int(vs[0]), int(vs[1])), 10, (0, 0, 255), -1)
        image_rgb = cv2.cvtColor(masked_region, cv2.COLOR_BGR2RGB)
        frame_filename = f"seg_out/frame_{str(frame_index).zfill(6)}_{dirg}.jpg"
        cv2.imwrite(frame_filename, image_rgb)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow("test", frame)
        frame_index += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()