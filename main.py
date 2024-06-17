'''
2024-05-23 작성자 : 장창호 
webcam_drowsiness_detection.py (v.3) 

1. 창호 노트북 환경에서 구현 완료
2. 현민 모델이름(best.pt)
3. requirements.txt 수정(pygame패키지 추가)
4. 졸음, 하품, 머리움직임에 대한 큐를 각각 별도로 생성해서 구현 BB는(Red, Yellow, Yellow)
5. 강한졸음, 약한졸음 -> 졸음으로 통합(800) / 인간의 평균 하품시간(6000) -> (testing 1000) / 고개 젖히기, 떨구기(800) 
6. 화면에도 특정조건에 만족하면서 졸음, 하품, 머리움직임에 대한 상태가 감지 되었을 때 경고 메세지 출력
7. 알람 파일 alarm.wav추가
8. 졸음 감지 -> 알람의 경우 졸음상태에서 벗어날때까지 알람 반복 재생(3초)
9. 하품, 머리움직임 감지 -> 하품, 머리움직임 상태가 감지되었을 때 알람 1번 울리고(1초) 초기화
'''

import cv2
import torch
import pygame
import numpy as np
from ultralytics import YOLO
from collections import deque
import datetime

# 상수 정의
FPS = 30  # 프레임
WARNING_DURATION = 2  # 경고 유지 시간(초)
QUEUE_DURATION = 2  # 큐에 저장할 시간(초)
YAWN_THRESHOLD_FRAMES = int(FPS * 1)  # 하품 상태 프레임 기준 -> 시연을 위해서 1초 정도로 변경 
# DROWSY_THRESHOLD_FRAMES = int(FPS * 0.4)  # 약한 졸음 상태 프레임 기준 -> 강한 졸음을 기준으로 하나의 '졸음'으로 정의
DROWSY_THRESHOLD_FRAMES = int(FPS * 0.8)  # 강한 졸음 상태 프레임 기준 -> 모든 sleep -> drowsy로 변경
HEAD_THRESHOLD_FRAMES = int(FPS * 0.8)  # 머리움직임 상태 프레임 기준

def play_alarm(sound_file, duration):
    # Pygame mixer 초기화
    pygame.mixer.init()
    # 사운드 파일 로드
    alarm_sound = pygame.mixer.Sound(sound_file)
    # 사운드 재생 (지정된 시간 동안)
    alarm_sound.play(loops=0, maxtime=duration)  # 8초짜리 알람데이터를 원하는대로 끊어서 반복 플레이시키기 위해 loop사용

def trigger_alarm(trigger, sound_file, duration):
    if trigger:
        print("알람이 울립니다!")
        play_alarm(sound_file, duration)
    else:
        print("알람이 울리지 않습니다.")

def get_webcam_fps():
    cap = cv2.VideoCapture(0)  # 웹캠을 엽니다.
    if not cap.isOpened():
        print("웹캠에 접근할 수 없습니다.")
        return None
    
    # FPS 가져오기
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 30

def load_model(model_path):
    # YOLOv8 모델을 로드합니다.
    model = YOLO(model_path)
    return model

def webcam_detection(model, fps):
    queue_length = int(fps * QUEUE_DURATION)
    # 한번 더 정의
    # drowsy_threshold_frames = int(fps * 0.4)  # 약한 졸음 -> 삭제
    drowsy_threshold_frames = int(fps * 0.8)  # 강한 졸음 -> 졸음
    yawn_threshold_frames = int(fps * 1)
    head_threshold_frames = int(fps * 0.8)
    
    eye_closed_queue = deque(maxlen=queue_length)
    yawn_queue = deque(maxlen=queue_length)
    head_queue = deque(maxlen=queue_length)
    head_warning_time = None
    yawn_warning_time = None
    drowsy_warning_time = None
    alarm_end_time = None

    cap = cv2.VideoCapture(0)  # 웹캠을 엽니다.
    if not cap.isOpened():
        print("웹캠에 접근할 수 없습니다.")
        return

    while True:  # 프레임마다 연산
        ret, frame = cap.read()  # 프레임을 읽습니다.
        if not ret:
            print("프레임을 가져오지 못했습니다.")
            break

        # 이미지를 전처리합니다.
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR에서 RGB로 변환합니다.
    
        results = model.predict(source=[img], save=False)[0]  # 모델 예측 결과는 리스트 형태로 반환 -> [0]: 첫 번째 결과

        # 결과를 시각화하고, 각 객체의 정보를 출력합니다.
        detected_event_list = []  # 감지된 이벤트를 저장하기 위한 빈 리스트를 초기화
        current_eye_closed = False
        current_yawn = False
        current_head_event = False

        for result in results:  # 감지된 모든 객체들을 반복 처리합니다.
            boxes = result.boxes  # 결과에서 바운딩 박스를 추출합니다.
            xyxy = boxes.xyxy.cpu().numpy()  # 바운딩 박스 좌표를 numpy 배열로 변환합니다.
            confs = boxes.conf.cpu().numpy()  # 신뢰도 점수를 numpy 배열로 변환합니다.
            classes = boxes.cls.cpu().numpy()  # 클래스 ID를 numpy 배열로 변환합니다.

            for i in range(len(xyxy)):  # xyxy 배열의 길이는 감지된 객체의 수와 동일 따라서, i는 각 객체의 인덱스를 의미
                xmin, ymin, xmax, ymax = map(int, xyxy[i])  # 바운딩 박스의 좌상단 좌표 (xmin, ymin)와 우하단 좌표 (xmax, ymax)를 정수형으로 추출
                confidence = confs[i]  # 현재 인덱스 i에 해당하는 객체의 신뢰도 점수를 가져옵니다. (confs 배열의 각 항목은 해당 객체가 실제로 존재할 확률을 나타냄)
                label = int(classes[i])  # 현재 인덱스 i에 해당하는 객체의 클래스 ID를 가져옵니다.
                
                # 객체 정보를 출력합니다.
                print(f"Detected {model.names[label]} with confidence {confidence:.2f} at [{xmin}, {ymin}, {xmax}, {ymax}]")

                if confidence > 0.5:  # 신뢰도가 0.5 이상인 객체만 표시
                    label_text = f"{model.names[label]} {confidence:.2f}"

                    # 기본 색상 설정 (초록색)
                    color = (0, 255, 0)

                    # 눈 감기 상태 확인 (label 0, 1, 2가 눈 감은 상태로 가정)
                    if label in [0, 1, 2]:
                        current_eye_closed = True

                    # 머리 숙임 또는 올림 상태 확인 (label 4, 5가 머리 상태)
                    if label in [4, 5]:
                        color = (0, 255, 255)  # 즉각적으로 노란색으로 설정
                        current_head_event = True

                    # 하품 상태 확인 (label 8)
                    if label == 8:
                        color = (0, 255, 255)  # 즉각적으로 노란색으로 설정
                        current_yawn = True

                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)  # 바운딩박스 그리기
                    cv2.putText(frame, label_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 현재 프레임의 눈 감기 상태를 큐에 추가
        eye_closed_queue.append(current_eye_closed)

        # 하품 이벤트 큐에 추가
        yawn_queue.append(current_yawn)

        # 머리 움직임 이벤트 큐에 추가
        head_queue.append(current_head_event)

        # 일정 기간 동안의 눈 감기 상태를 기반으로 졸음 및 수면 상태 판별
        eye_closed_count = sum(eye_closed_queue)
        if eye_closed_count >= drowsy_threshold_frames:
            detected_event_list.append('drowsy')
            drowsy_warning_time = datetime.datetime.now()
            if alarm_end_time is None or datetime.datetime.now() >= alarm_end_time:
                trigger_alarm(True, 'alarm.wav', 3000)  # 3초 동안 알람
                alarm_end_time = datetime.datetime.now() + datetime.timedelta(seconds=3)

        # # 약한 졸음 상태는 주석 처리
        # elif eye_closed_count >= drowsy_threshold_frames:
        #     detected_event_list.append('drowsy')
        #     drowsy_warning_time = datetime.datetime.now()
        #     if alarm_end_time is None or datetime.datetime.now() >= alarm_end_time:
        #         trigger_alarm(True, 'alarm.wav', 1000)  # 1초 동안 알람
        #         alarm_end_time = datetime.datetime.now() + datetime.timedelta(seconds=1)

        # 하품이 일정 횟수 이상 발생하면 경고
        yawn_count = sum(yawn_queue)
        if yawn_count >= yawn_threshold_frames:
            detected_event_list.append('yawn')
            yawn_warning_time = datetime.datetime.now()
            if alarm_end_time is None or datetime.datetime.now() >= alarm_end_time:
                trigger_alarm(True, 'alarm.wav', 1000)  # 1초 동안 알람
                alarm_end_time = datetime.datetime.now() + datetime.timedelta(seconds=1)
            yawn_queue.clear()  # 하품 상태 초기화

        # 머리 움직임이 일정 횟수 이상 발생하면 경고
        head_event_count = sum(head_queue)
        if head_event_count >= head_threshold_frames:
            detected_event_list.append('head_movement')
            head_warning_time = datetime.datetime.now()
            if alarm_end_time is None or datetime.datetime.now() >= alarm_end_time:
                trigger_alarm(True, 'alarm.wav', 1000)  # 1초 동안 알람
                alarm_end_time = datetime.datetime.now() + datetime.timedelta(seconds=1)
            head_queue.clear()  # 머리 움직임 상태 초기화

        if eye_closed_count < drowsy_threshold_frames and yawn_count < yawn_threshold_frames and head_event_count < head_threshold_frames:
            alarm_end_time = None

        # 현재 시간
        current_time = datetime.datetime.now()

        # 눈 감기 상태에 따른 색상 변경
        for result in results:
            boxes = result.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            classes = boxes.cls.cpu().numpy()

            for i in range(len(xyxy)):
                xmin, ymin, xmax, ymax = map(int, xyxy[i])
                label = int(classes[i])
                if label in [0, 1, 2]:  # 눈 감기 상태일 때만 색상 변경
                    if 'drowsy' in detected_event_list:
                        color = (0, 0, 255)  # 빨간색
                    # elif 'drowsy' in detected_event_list:
                    #     color = (0, 165, 255)  # 주황색
                    else:
                        color = (0, 255, 0)  # 초록색

                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                    cv2.putText(frame, model.names[label], (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 워닝 메시지 표시
        font_scale = 0.75  # 폰트 크기 줄임
        font_thickness = 2  # 폰트 두께 줄임

        # 텍스트 위치 : 왼쪽 상단 모서리를 원점 (0, 0)으로 하고, x 좌표는 오른쪽으로, y 좌표는 아래쪽으로 증가
        if drowsy_warning_time and (current_time - drowsy_warning_time).total_seconds() < WARNING_DURATION:
            cv2.putText(frame, 'Warning: Drowsy Detected!', (50, 150), cv2.FONT_ITALIC, font_scale, (0, 0, 255), font_thickness)
        # elif drowsy_warning_time and (current_time - drowsy_warning_time).total_seconds() < WARNING_DURATION:
        #     cv2.putText(frame, 'Warning: Drowsy Detected!', (50, 150), cv2.FONT_ITALIC, font_scale, (0, 165, 255), font_thickness)
        if yawn_warning_time and (current_time - yawn_warning_time).total_seconds() < WARNING_DURATION:
            cv2.putText(frame, 'Warning: Yawning Detected!', (50, 50), cv2.FONT_ITALIC, font_scale, (0, 255, 255), font_thickness)
        if head_warning_time and (current_time - head_warning_time).total_seconds() < WARNING_DURATION:
            cv2.putText(frame, 'Warning: Head Up/Down Detected!', (50, 100), cv2.FONT_ITALIC, font_scale, (0, 255, 255), font_thickness)

        cv2.imshow('YOLOv8 Webcam Object Detection', frame)  # 결과를 화면에 표시합니다.
        if cv2.waitKey(1) == ord('q'):  # 'q' 키를 누르면 종료합니다.
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    fps = get_webcam_fps()
    print(f"웹캠 프레임 수: {fps} FPS")
    model_path = 'best_2nd.pt'  # 모델 파일 경로를 설정합니다.
    model = load_model(model_path)
    webcam_detection(model, fps)
