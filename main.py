import cv2
from SafeZone import Safe_Zone 
import argparse

def extract_frames(video_path, output_folder, frame_rate=1):
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_path)

    # 프레임 수 정보
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate_actual = int(cap.get(cv2.CAP_PROP_FPS))

    # 요청된 프레임레이트보다 낮으면 오류 반환
    if frame_rate_actual < frame_rate:
        cap.release()
        raise ValueError("The frame rate of the video is lower than the requested frame rate.")

    # 프레임 인덱스 초기화
    frame_index = 0

    while True:
        # 프레임 읽기
        ret, frame = cap.read()

        # 비디오의 끝에 도달하면 반복 종료
        if not ret:
            break
        
        try:
            safe = Safe_Zone(frame)
            save_mask, masks2 = safe.SAM()
            #vs = safe.VanishingPoint(save_mask, masks2)
            vs = safe.VanishingPoint(frame)

            if vs is None:
                vs = vs_previous
            else:
                vs_previous = vs

            frame = safe.Angular_Bisector(masks2,vs)

            cv2.circle(save_mask, (int(vs[0]), int(vs[1])), 50, (0, 0, 255), -1)
            image_rgb = cv2.cvtColor(save_mask, cv2.COLOR_BGR2RGB)
            frame_filename = f"seg_out/frame_{str(frame_index).zfill(6)}.jpg"
            cv2.imwrite(frame_filename, image_rgb)

            # 요청된 프레임레이트로 프레임 저장
            #if frame_index % (frame_rate_actual // frame_rate) == 0:
                # 이미지로 저장 (파일명 예시: frame_000001.jpg)
            frame_filename = f"{output_folder}/frame_{str(frame_index).zfill(6)}.jpg"
            cv2.imwrite(frame_filename, frame)
        except:
            pass

        frame_index += 1

    # 캡처 객체 해제
    cap.release()

    print(f"Total frames in the video: {total_frames}")
    print(f"Extracted frames at {frame_rate}fps and saved to '{output_folder}'")


# 5fps로 이미지 추출
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path')
    parser.add_argument('--output_folder')
    parser.add_argument('--frame_rate',default=5,type=int)

    args = parser.parse_args()

    video_path = args.video_path
    output_folder = args.output_folder
    frame_rate = args.frame_rate

    extract_frames(video_path, output_folder, frame_rate=frame_rate)

# main 실행
if __name__ == "__main__":
    main()
