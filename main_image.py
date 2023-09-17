import cv2
import numpy as np
import time

from SafeZone import Safe_Zone

def extract_masked_region(image, mask):
    # Mask 값이 1인 픽셀만 추출하여 새로운 이미지 생성
    masked_image = np.copy(image)
    masked_image[mask != 1] = 0

    return masked_image

# 이미지 파일 경로 설정
image_path = "./test.jpg"
output_dir = "./seg_out/"  # 결과 이미지를 저장할 디렉토리를 수정하세요

# 이미지 변경 간격 설정 (초당 10번 이미지 변경)
change_interval = 1 / 100.0

while True:
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Could not read image from {image_path}")
        continue

    safe = Safe_Zone(frame)
    save_mask, masks2 = safe.SAM()

    vs = safe.VanishingPoint_Triangle(masks2)

    if vs is None or vs[0] == 0 or vs[1] == 0:
        try:
            vs = vs_previous
        except:
            vs = [int(frame.shape[0] / 2), int(frame.shape[1] / 2)]
    else:
        vs_previous = vs

    frame, pr_mask, pr_x1, pr_x2 = safe.Angular_Bisector(masks2, vs)

    masks_save = np.expand_dims(masks2, axis=2)
    masks_save = np.squeeze(masks_save)
    masked_region = extract_masked_region(frame, masks_save)

    if frame.shape[1] / 2 < pr_x1:
        print('right')
        dirg = 'right'
    elif frame.shape[1] / 2 > pr_x2:
        print('left')
        dirg = 'left'
    else:
        print('Normal')
        dirg = 'Normal'

    cv2.circle(masked_region, (int(vs[0]), int(vs[1])), 10, (0, 0, 255), -1)
    image_rgb = cv2.cvtColor(masked_region, cv2.COLOR_BGR2RGB)
    frame_filename = f"{output_dir}/{dirg}.jpg"
    cv2.imwrite(frame_filename, image_rgb)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow("test", frame)
    cv2.waitKey(1)  # 이미지가 빠르게 변경되는 경우, 각 이미지를 표시하기 위해 짧은 대기 시간을 추가

    # 이미지 변경 간격을 지켜서 코드 실행
    # time.sleep(change_interval)

cv2.destroyAllWindows()
