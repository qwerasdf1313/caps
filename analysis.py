import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
from PIL import ImageDraw, ImageFont, Image, ImageColor
import tempfile
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas  # Import FigureCanvas
import plotly.express as px
import pandas as pd

# Mediapipe 모델 파일 경로를 환경 변수로 설정
os.environ["MEDIAPIPE_MODEL_PATH"] = "/tmp/mediapipe_models"

# 쓰기 가능한 디렉토리 생성
os.makedirs("/tmp/mediapipe_models", exist_ok=True)

# Mediapipe Pose 사용 예시
with mp.solutions.pose.Pose(
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    static_image_mode=False
) as pose:
    pass

# Mediapipe initialization
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 상태 초기화
if "results_ready" not in st.session_state:
    st.session_state["results_ready"] = False
    st.session_state["max_reba_score"] = 0
    st.session_state["max_frame"] = None
    st.session_state["max_reba_scores"] = None


# Initialize the scores dictionary
scores = {
    'Reba': 0,
    'Trunk': 0,
    'Neck': 0,
    'Leg': 0,
    'Upper_arm': 0,
    'Down_arm': 0,
    'Wrist': 0
}

SCORES_COLORS = {
    "Trunk": ["#00FF00", "#A4FF00", "#FFFF00", "#FFA500", "#FF0000"],
    "Neck": ["#00FF00", "#FFA500", "#FF0000"],
    "Leg": ["#00FF00", "#FFFF00", "#FFA500", "#FF0000"],
    "Upper Arm": ["#00FF00", "#A4FF00", "#FFFF00", "#FFA500", "#FF0000", "#B22222"],
    "Down Arm": ["#00FF00", "#FF0000"],
    "Wrist": ["#00FF00", "#FFA500", "#FF0000"],
}

# Plotly를 사용한 REBA 점수 시각화 함수
def plot_reba_scores_plotly(scores):
    # 올바른 데이터 형태로 변환
    labels = ['REBA', 'Trunk', 'Neck', 'Leg', 'Upper Arm', 'Down Arm', 'Wrist']
    scores_values = list(scores)  # scores가 리스트로 변환되었는지 확인

    # 데이터 생성
    data = {'Category': labels, 'Score': scores_values}
    df = pd.DataFrame(data)
    
    # Plotly 그래프 생성
    fig = px.bar(
        data,
        x='Category',
        y='Score',
        title="REBA Scores",
        color='Score',
        color_continuous_scale=px.colors.sequential.Viridis,
        labels={'Score': 'Score (0-15)', 'Category': 'Body Part'}
    )
    fig.update_layout(
        xaxis_title="Body Parts",
        yaxis_title="Score",
        yaxis=dict(range=[0, 15]),
        template='plotly_white',
        width=800,
        height=400
    )
    return fig

# Function to process the image
def process_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process with pose model
    pose_results = pose.process(image_rgb)
    
    # Process with hand model
    hands_results = hands.process(image_rgb)
    
    annotated_image = image.copy()
    
    if pose_results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )
        
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )
    
    return annotated_image, pose_results.pose_landmarks, hands_results.multi_hand_landmarks if hands_results.multi_hand_landmarks else None


def plot_reba_scores(scores):
    labels = ['REBA', 'Trunk', 'Neck', 'Leg', 'Upper Arm', 'Down Arm', 'Wrist']
    plt.figure(figsize=(10, 6))
    plt.bar(labels, scores, color='skyblue')
    plt.title("REBA Scores")
    plt.ylim(0, 15)
    st.pyplot(plt)

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def calculate_angle2(a, b):
    a = np.array(a)
    b = np.array(b)
    radians = np.arctan2(b[1] - a[1], b[0] - a[0])
    angle = np.degrees(radians)
    return angle


# Function to evaluate REBA (placeholder)
def evaluate_reba(landmarks):
    left_shoulder = [
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
            ]
    left_shoulder_visibility = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility
    left_hip = [
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
               landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
            ]
    left_hip_visibility = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility
    left_elbow = [
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y
            ]
    left_elbow_visibility = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility
    left_wrist = [
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y

            ]

    left_knee = [
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
            ]
    left_ankle = [
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
            ]

            # Get coordinates and visibility for right arm
    right_shoulder = [
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
            ]
    right_shoulder_visibility = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility
    right_hip = [
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
            ]
    right_hip_visibility = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility
    right_elbow = [
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y
            ]
    right_elbow_visibility = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility
    right_wrist = [
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x ,
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y 
            ]
    right_knee = [
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y
            ]
    right_ankle = [
                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
            ]

    # Get coordinates for ears
    left_ear = [
                landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y
            ]
    right_ear = [
                landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y
            ]
    left_index = [
                landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y
            ]
    right_index = [
                landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y
            ]    

    midpoint_shoulder = [(left_shoulder[0]+right_shoulder[0]) / 2,(left_shoulder[1]+right_shoulder[1]) / 2]

    head = [(left_ear[0]+right_ear[0]) / 2,(left_ear[1]+right_ear[1]) / 2]

    midpoint_hip = [(left_hip[0]+right_hip[0]) / 2,(left_hip[1]+right_hip[1]) / 2]

    trunk_angle = calculate_angle2(midpoint_shoulder, midpoint_hip)

    trunk_score = 0
    if trunk_angle == 90:
        trunk_score = 1
    elif 70 < trunk_angle < 90:
        trunk_score = 2
    elif 90 < trunk_angle <= 110:
        trunk_score = 2
    elif 30 < trunk_angle <= 70:
        trunk_score = 3
    elif trunk_angle > 110:
        trunk_score = 3
    elif trunk_angle <= 30:
        trunk_score = 4

    neck_angle = calculate_angle(head, midpoint_shoulder, midpoint_hip)

    neck_score = 0
    if 160 < neck_angle <= 180:
        neck_score = 1
    elif neck_angle <= 160:
        neck_score = 2


    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

    leg_score = 0

    if abs(left_knee_angle - right_knee_angle) > 10:
        leg_score += 2
    else:
        leg_score += 1

    if (120 < left_knee_angle <= 150) or (120 <= right_knee_angle < 150):
        leg_score += 1
    elif (left_knee_angle <= 120) or (right_knee_angle <= 120):
        leg_score += 2

    leg_score = 1

    reba_table_a = [[[1,2,3,4],[1,2,3,4],[3,3,5,6]],[[2,3,4,5],[3,4,5,6],[4,5,6,7]],[[2,4,5,6],[4,5,6,7],[5,6,7,8]],[[3,5,6,7],[5,6,7,8],[6,7,8,9]],[[4,6,7,8],[6,7,8,9],[7,8,9,9]]]

    # Get the Group A score from the lookup table
    group_a_score = reba_table_a[trunk_score-1][neck_score-1][leg_score-1]

    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    upperarm_left = calculate_angle(left_elbow, left_shoulder, left_hip)
    upperarm_right = calculate_angle(right_elbow, right_shoulder, right_hip)

    upperarm_left_angle = calculate_angle(left_elbow,left_shoulder,left_hip)
    upperarm_right_angle = calculate_angle(right_elbow, right_shoulder, right_hip)
    downarm_left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    downarm_right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    wrist_right_agle = calculate_angle(right_elbow, right_wrist, right_index)
    wrist_left_agle = calculate_angle(left_elbow, left_wrist, left_index)

    upper_arm_score = 0
    
    if (0 <= upperarm_left_angle < 20) or (0 <= upperarm_right_angle < 20):
        upper_arm_score = 1
    elif (20 <= upperarm_left_angle < 45) or (20 <= upperarm_right_angle < 45):
        upper_arm_score = 2
    elif (45 <= upperarm_left_angle < 90) or (45 <= upperarm_right_angle < 90):
        upper_arm_score = 3
    else:
        upper_arm_score = 4
        
    down_arm_score = 0
    
    if (80 < downarm_left_angle < 120) or (80 < downarm_right_angle < 120):
        down_arm_score = 1
    else:
        down_arm_score = 2

    wrist_score = 0
    
    if ( 165 < wrist_left_agle < 195) or ( 165 < wrist_right_agle < 195):
        wrist_score = 1
    else:
        wrist_score = 2

    reba_table_b = [
    [
        [1, 2, 2],
        [1, 2, 3]
    ],
    [
        [1, 2, 3],
        [2, 3, 4]
    ],
    [
        [3, 4, 5],
        [4, 5, 5]
    ],
    [
        [4, 5, 6],
        [5, 6, 7]
    ],
    [
        [6, 7, 8],
        [7, 8, 8]
    ],
    [
        [7, 8, 8],
        [8, 9, 9]
    ]
]

    group_b_score = reba_table_b[upper_arm_score-1][down_arm_score-1][wrist_score-1]

    reba_table_c = [
    [1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7],
    [1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8],
    [2, 3, 3, 3, 4, 5, 6, 7, 7, 8, 8, 8],
    [3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9],
    [4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9, 9],
    [6, 6, 6, 7, 8, 8, 9, 9, 10, 10, 10, 10],
    [7, 7, 7, 8, 9, 9, 9, 10, 10, 11, 11, 11],
    [8, 8, 8, 9, 10, 10, 10, 10, 10, 11, 11, 11],
    [9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12],
    [10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12],
    [11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12],
    [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]
]
    reba_score = reba_table_c[group_a_score][group_b_score]

    return reba_score, trunk_score, neck_score, leg_score, upper_arm_score, down_arm_score, wrist_score

def annotate_frame_with_reba(frame, scores, overall_score, landmarks):
    """
    동영상 프레임에 REBA 점수 및 관절별 색상 표시
    """
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # OpenCV → PIL 변환
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.load_default()  # 기본 폰트 사용

    # REBA 점수 및 위험 수준 표시
    risk_text, risk_color = get_risk_level(overall_score)
    draw.text((10, 10), f"REBA Score: {overall_score}", fill="blue", font=font)
    draw.text((10, 30), f"Risk Level: {risk_text}", fill=risk_color, font=font)

    # 관절별 위치 계산 및 점수 표시
    positions = {
        "Trunk": [(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x + landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x) / 2,
                  (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y + landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y) / 2],
        "Neck": [(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x + landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x) / 2,
                 landmarks[mp_pose.PoseLandmark.NOSE.value].y],
        "Leg": [(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x + landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x) / 2,
                (landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y + landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y) / 2],
        "Upper Arm": [(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x) / 2,
                      (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y) / 2],
        "Down Arm": [(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x) / 2,
                     (landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y) / 2],
        "Wrist": [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y],
    }

    # 관절별 점수 및 색상 표시
    for i, (part, pos) in enumerate(positions.items()):
        score = scores[i + 1]  # REBA 점수 제외
        color_hex = SCORES_COLORS[part][min(score - 1, len(SCORES_COLORS[part]) - 1)]  # 점수에 따른 색상
        color_rgb = ImageColor.getrgb(color_hex)  # HEX → RGB 변환
        x, y = int(pos[0] * frame.shape[1]), int(pos[1] * frame.shape[0])  # 정규화된 좌표 → 실제 좌표
        draw.ellipse([(x - 10, y - 10), (x + 10, y + 10)], fill=color_rgb)  # 원형으로 표시
        draw.text((x + 15, y - 5), f"{part}: {score}", fill="black", font=font)

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)  # PIL → OpenCV 변환

# 위험성 표시
def get_risk_level(reba_score):
    if reba_score == 1:
        return "위험성 낮음", "#00FF00"
    elif 2 <= reba_score <= 3:
        return "위험성 조금 있음", "#A4FF00"
    elif 4 <= reba_score <= 7:
        return "위험성 있음", "#FFA500"
    elif 8 <= reba_score <= 10:
        return "위험성 높음", "#FF4500"
    elif 11 <= reba_score <= 15:
        return "위험성 매우 높음", "#FF0000"

def annotate_frame_with_reba(frame, scores, overall_score, landmarks):
    """
    동영상 프레임에 REBA 점수 및 관절별 색상 표시
    """
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # OpenCV → PIL 변환
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.load_default()  # 기본 폰트 사용

    # REBA 점수 및 위험 수준 표시
    risk_text, risk_color = get_risk_level(overall_score)
    draw.text((10, 10), f"REBA Score: {overall_score}", fill="blue", font=font)
    draw.text((10, 30), f"Risk Level: {risk_text}", fill=risk_color, font=font)

    # 관절별 위치 계산 및 점수 표시
    positions = {
        "Trunk": [(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x + landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x) / 2,
                  (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y + landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y) / 2],
        "Neck": [(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x + landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x) / 2,
                 landmarks[mp_pose.PoseLandmark.NOSE.value].y],
        "Leg": [(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x + landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x) / 2,
                (landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y + landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y) / 2],
        "Upper Arm": [(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x) / 2,
                      (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y) / 2],
        "Down Arm": [(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x) / 2,
                     (landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y) / 2],
        "Wrist": [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y],
    }

    # 관절별 점수 및 색상 표시
    for i, (part, pos) in enumerate(positions.items()):
        score = scores[i + 1]  # REBA 점수 제외
        color_hex = SCORES_COLORS[part][min(score - 1, len(SCORES_COLORS[part]) - 1)]  # 점수에 따른 색상
        color_rgb = ImageColor.getrgb(color_hex)  # HEX → RGB 변환
        x, y = int(pos[0] * frame.shape[1]), int(pos[1] * frame.shape[0])  # 정규화된 좌표 → 실제 좌표
        draw.ellipse([(x - 10, y - 10), (x + 10, y + 10)], fill=color_rgb)  # 원형으로 표시
        draw.text((x + 15, y - 5), f"{part}: {score}", fill="black", font=font)

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)  # PIL → OpenCV 변환


def annotate_body_image(image_path, scores, overall_score):
    # 이미지 로드 및 RGB 모드로 변환
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    # 각 부위 점수 및 위치
    body_parts = ["Trunk", "Neck", "Leg", "Upper Arm", "Down Arm", "Wrist"]
    positions = [(120, 50), (120, 30), (120, 300), (80, 150), (160, 150), (120, 220)]

    # 텍스트 추가
    for i, (score, pos) in enumerate(zip(scores[1:], positions)):
        color = (255, 0, 0) if score > 3 else (0, 255, 0)  # 빨강/초록 색상
        draw.text(pos, f"{body_parts[i]}: {score}", fill=color, font=font)

    # REBA 점수 및 위험 수준
    draw.text((10, 10), f"REBA Score: {overall_score}", fill=(0, 0, 255), font=font)
    return img


# Streamlit app interface
st.title('REBA Evaluation Web App')
st.write("Upload an image or video and get a REBA evaluation.")


FRAME_WINDOW = st.empty()

def analysis_page():
    st.title("Real-Time REBA Evaluation")

    # 상태 초기화
    if "max_reba_score" not in st.session_state:
        st.session_state["max_reba_score"] = 0
        st.session_state["max_frame"] = None
        st.session_state["max_reba_scores"] = None

    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()  # 실시간 영상 출력
        stgraph = st.empty()  # 실시간 그래프 출력

        with mp_pose.Pose(model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5, static_image_mode=False) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = frame[:, :int(frame.shape[1] * 0.5)]
                
                # Mediapipe 포즈 추출
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    landmarks = results.pose_landmarks.landmark

                    reba_score, trunk_score, neck_score, leg_score, upper_arm_score, down_arm_score, wrist_score = evaluate_reba(landmarks)

                    # REBA 점수 업데이트
                    updated_scores = [reba_score, trunk_score, neck_score, leg_score, upper_arm_score, down_arm_score, wrist_score]

                    # 최대 REBA 점수 업데이트
                    if reba_score > st.session_state["max_reba_score"]:
                        st.session_state["max_reba_score"] = reba_score
                        st.session_state["max_frame"] = frame.copy()
                        st.session_state["max_reba_scores"] = updated_scores
                        st.session_state["landmarks"] = landmarks  # 랜드마크 저장

                # Plotly 그래프 생성 및 업데이트
                    updated_figure = plot_reba_scores_plotly(updated_scores)
                
                    stframe.image(frame, channels="BGR", caption="Video Frame")

 # Plotly 그래프 업데이트
                    with stgraph.container():
                        st.plotly_chart(updated_figure, use_container_width=True)
            

        cap.release()

        # 분석 결과 출력
        stframe.empty()  # 동영상 자리 비우기
        stgraph.empty()
        if st.session_state["max_frame"] is not None and st.session_state["landmarks"] is not None:
            # 관절별 결과를 프레임에 표시
            scores = st.session_state["max_reba_scores"]
            overall_score = st.session_state["max_reba_score"]
            landmarks = st.session_state["landmarks"]
            annotated_frame = annotate_frame_with_reba(st.session_state["max_frame"], scores, overall_score, landmarks)
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # 결과 출력
            stframe.image(annotated_frame_rgb, caption="Frame with Highest REBA Score", use_column_width=True)

            # 위험성 표시
            risk_text, risk_color = get_risk_level(overall_score)
            st.markdown(f"### REBA 점수: {overall_score}")
            st.markdown(f"#### 위험 수준: <span style='color:{risk_color}'>{risk_text}</span>", unsafe_allow_html=True)
        else:
            st.warning("No poses detected during the video.")

if __name__ == "__main__":
    analysis_page()
