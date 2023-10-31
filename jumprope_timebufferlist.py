import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import time

class BufferList:
    def __init__(self, buffer_time, default_value=0):
        self.buffer = [default_value for _ in range(buffer_time)]

    def push(self, value):
        self.buffer.pop(0)
        self.buffer.append(value)

    def max(self):
        return max(self.buffer)

    def min(self):
        buffer = [value for value in self.buffer if value]
        if buffer:
            return min(buffer)
        return 0

class BufferList2:
    def __init__(self, buffer_time_jump, default_value=None):
        self.buffer = [default_value for _ in range(buffer_time_jump)]
        
    def push(self, value):
        if self.buffer[0] is None:
            self.buffer[0] = time.time()

        current_time = time.time()
        time_difference = current_time - self.buffer[-1]

        self.buffer.pop(0)
        self.buffer.append(value)
        
        return time_difference

    def max(self):
        return max(self.buffer)

    def min(self):
        buffer = [value for value in self.buffer if value]
        if buffer:
            return min(buffer)
        return 0


# test file
file_name = "cw0_output.mp4"
#   mda-kc1e2xm7b3t9vzgt.mp4      1foot.gif     1foot2.gif      1foot.mp4    1foot2.mp4     jumprope1.gif

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# center y
selected_landmarks = [23, 24]




#버퍼크기 buffer_time 크면 center값 버퍼 채우기 위해 시작시 딜레이 생김.
buffer_time = 5
buffer_time_jump = 1
center_y = BufferList(buffer_time)
center_y_up = BufferList(buffer_time)
center_y_down = BufferList(buffer_time)
center_y_pref_flip = BufferList(buffer_time)
center_y_flip = BufferList(buffer_time)
center_y_shoulder_hip = BufferList(buffer_time)
time_difference = BufferList2(buffer_time_jump)


cy_max = 300
cy_min = 300
flip_flag = 260
prev_flip_flag = 260
count = 0
td = 0
f_td = 0

# webcam input:
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(file_name)

# 원하는 해상도 설정 (640x480)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# 비디오 파일의 FPS 값 읽어오기
fps = int(cap.get(cv2.CAP_PROP_FPS))
# fps = 30


# 비디오의 원래 FPS에 맞게 프레임 딜레이 설정
frame_delay = int(1000 / fps)  # 밀리초 단위 딜레이




fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    file_name.replace(".mp4", "_output.mp4"),
    fourcc,
    float(fps),
    # 25.0,
    (int(cap.get(3)), int(cap.get(4))),
)



with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image_height, image_width, _ = image.shape
        # # 카메라 회전 고려해서 height>width일 경우 그냥 진행 >일경우 둘을 바꿔줌 
        # if image_height > image_width: 
        #     rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        #     # 회전된 이미지의 높이와 너비를 업데이트
        #     rotated_height, rotated_width, _ = rotated_image.shape
        #     image_width = rotated_width
        #     image_height = rotated_height


        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )
            landmarks = [
                (lm.x * image_width, lm.y * image_height)
                for i, lm in enumerate(results.pose_landmarks.landmark)
                if i in selected_landmarks
            ]
            cx = int(np.mean([x[0] for x in landmarks]))
            cy = int(np.mean([x[1] for x in landmarks]))

            landmarks = [
                (lm.x * image_width, lm.y * image_height)
                for i, lm in enumerate(results.pose_landmarks.landmark)
                if i in [11, 12]
            ]
            cy_shoulder_hip = cy - int(np.mean([x[1] for x in landmarks]))
        else:
            #대략적인 중심점 위치
            # cx = int(image_width*0.5)
            # cy = int(image_height*0.56)
            # cy_shoulder_hip = int(image_height*0.28)
            cx = int(image_width)
            cy = int(image_height)
            cy_shoulder_hip = int(image_height)

        center_y_shoulder_hip.push(cy_shoulder_hip)

        cy = int((cy + center_y.buffer[-1]) / 2)
        # set data
        center_y.push(cy)


        cy_max = 0.5 * cy_max + 0.5 * center_y.max()
        center_y_up.push(cy_max) #그래프 그릴때 필요
    
        cy_min = 0.5 * cy_min + 0.5 * center_y.min()
        center_y_down.push(cy_min) #그래프 그릴때 필요

        prev_flip_flag = flip_flag
        center_y_pref_flip.push(prev_flip_flag) #그래프 그릴때 필요


        #현재시간 측정
        current_time = time.time()

        #점프 감지
        dy = cy_max - cy_min    #dy 25
        if dy > 0.1 * cy_shoulder_hip:  #점프 민감도. 0.4로 설정시 7번 이후엔 측정 안되는경우가 있음. cy_shoulder=80~90  0.2로 낮춤
            # if cy > cy_max - 0.55 * dy and flip_flag == 150:   #최저점 cymax = 최저높이     
            #     flip_flag = 250
            # if 0 < cy < cy_min + 0.35 * dy and flip_flag == 250:   #최고점 cymin = 최고높이    
            #     flip_flag = 150
            if cy > cy_max - 0.35 * dy and flip_flag == 250:   #최저점 cymax = 최저높이     320 - 14   306 300  
                flip_flag = 260
            if 0 < cy < cy_min + 0.25 * dy and flip_flag == 260:   #최고점 cymin = 최고높이    295 + 9   304 310
                flip_flag = 250




            # if cy > cy_max - 0.2 * dy and flip_flag == 250:   #최저점 cymax = 최저높이     320 - 14   306 300  
            #     flip_flag = 260
            # if 0 < cy < cy_min + 0.2 * dy and flip_flag == 260:   #최고점 cymin = 최고높이    295 + 9   304 310
            #     flip_flag = 250


            # if cy > cy_max - 0.72 * dy and flip_flag == 150:   #최저점 cymax = 최저높이
            #     flip_flag = 250
            # if 0 < cy < cy_min + 0.46 * dy and flip_flag == 250:   #최고점 cymin = 최고높이
            #     flip_flag = 150


                # .push()를 호출하여 시간 차이를 얻고 변수에 저장
                td = time_difference.push(current_time)
                f_td = f"{td:0.2f}"
        center_y_flip.push(flip_flag) #그래프 그릴때 필요



        # if prev_flip_flag < flip_flag and td > 0.6 and td < 2.0:
        if count != 0 and prev_flip_flag < flip_flag and td > 0.1:
            count += 1

        # if prev_flip_flag < flip_flag:  # td 시간 차이가 0.6 이상 2.0미만인 경우 카운트+
        if count == 0 and prev_flip_flag < flip_flag:
            count = 1

        
        cv2.line(image, (int(0.3*image_width), int(0.1*image_height)), (int(0.7*image_width), int(0.1*image_height)), (0, 255, 0))
        cv2.putText(
            image,
            "Head line",
            (int(0.72*image_width), int(0.11*image_height)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        cv2.line(image, (int(0.3*image_width), int(0.95*image_height)), (int(0.7*image_width), int(0.95*image_height)), (0, 255, 0))
        cv2.putText(
            image,
            "Foot line",
            (int(0.72*image_width), int(0.96*image_height)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(
            image,
            "Centroid",
            (cx - 25, cy - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
        cv2.putText(
            image,
            "Count:" + str(count),
            (int(image_width * 0.55), int(image_height * 0.4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            1,
        )
        # cv2.putText(
        #     image,
        #     "FPS:" + str(fps),
        #     (int(image_width * 0.05), int(image_height * 0.4)),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5,
        #     (0, 0, 255),
        #     1,
        # )
        cv2.putText(
            image,
            "Jump delay:" + str(f_td),
            (int(image_width * 0.05), int(image_height * 0.5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )

        # plt.clf()
        # plt.plot(center_y.buffer, label="center_y")
        # plt.plot(center_y_up.buffer, label="center_y_up")
        # plt.plot(center_y_down.buffer, label="center_y_down")
        # plt.plot(center_y_pref_flip.buffer, label="center_y_pref_flip")
        # plt.plot(center_y_flip.buffer, label="center_y_flip")
        # plt.plot(center_y_shoulder_hip.buffer, label="center_y_shoulder_hip")
        # plt.legend(loc="upper right")
        # plt.pause(0.1)

        # display.
        cv2.imshow("Jumprope Check", image)
        # out.write(image)
        if cv2.waitKey(frame_delay) & 0xFF == 27:
            break


cap.release()
out.release()
cv2.destroyAllWindows()

