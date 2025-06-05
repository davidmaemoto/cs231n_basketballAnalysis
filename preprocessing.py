import os
import cv2
import csv

# CONFIG
VIDEO_DIR = './data/shot_vids'
OUTPUT_DIR = './data/shot_timestamps'
ROI_RATIO = (0.5, 0.5)
MINIMUM_FRAMES_NEEDED = 10
HOUGH_DP = 1.2
HOUGH_MIN_DIST = 20
HOUGH_PARAM1 = 50
HOUGH_PARAM2 = 30
HOUGH_MIN_RADIUS = 5
HOUGH_MAX_RADIUS = 50
NEW_SHOT_ENTRY_X_FRAC = 0.3
DISPLAY = True

os.makedirs(OUTPUT_DIR, exist_ok=True)

def detect_ball(cropped_gray, fgmask):
    masked = cv2.bitwise_and(cropped_gray, cropped_gray, mask=fgmask)
    circles = cv2.HoughCircles(
        masked,
        cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=HOUGH_MIN_DIST,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=HOUGH_MIN_RADIUS,
        maxRadius=HOUGH_MAX_RADIUS
    )
    if circles is not None:
        x, y, r = map(int, circles[0][0])
        return x, y, r
    return None

for fname in os.listdir(VIDEO_DIR):
    video_path = os.path.join(VIDEO_DIR, fname)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    roi_w = int(width * ROI_RATIO[0])
    roi_h = int(height * ROI_RATIO[1])
    subtractor = cv2.createBackgroundSubtractorMOG2(history=500,
                                                   varThreshold=16,
                                                   detectShadows=False)

    timestamps = []
    in_shot = False
    start_frame = None
    no_detect_count = 0
    frame_idx = 0

    if DISPLAY:
        cv2.namedWindow('Shot Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Shot Detection', width, height)

    while True:
        ret, frame = cap.read()
        if not ret:
            if in_shot:
                end_frame = frame_idx - 1
                timestamps.append((start_frame / fps, end_frame / fps))
            break
        roi = frame[0:roi_h, 0:roi_w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        fgmask = subtractor.apply(roi)
        _, fgmask = cv2.threshold(fgmask, 254, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)
        circle = detect_ball(blurred, fgmask)
        ball_present = False
        if circle is not None:
            x, y, r = circle
            if not in_shot:
                if x < roi_w * NEW_SHOT_ENTRY_X_FRAC:
                    ball_present = True
            else:
                ball_present = True
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (roi_w, roi_h), (255, 0, 0), 2)
        if ball_present:
            cv2.circle(overlay, (x, y), r, (0, 255, 0), 2)
        if ball_present and not in_shot:
            in_shot = True
            start_frame = frame_idx
            no_detect_count = 0
            cv2.putText(overlay, 'SHOT START', (50, height - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif in_shot and not ball_present:
            no_detect_count += 1
            if no_detect_count >= MINIMUM_FRAMES_NEEDED:
                end_frame = frame_idx - no_detect_count
                timestamps.append((start_frame / fps, end_frame / fps))
                in_shot = False
                start_frame = None
                no_detect_count = 0
                cv2.putText(overlay, 'SHOT END', (50, height - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        time_sec = frame_idx / fps
        cv2.putText(overlay, f'{time_sec:.2f}s', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if DISPLAY:
            cv2.imshow('Shot Detection', overlay)
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                break
        frame_idx += 1

    cap.release()
    if DISPLAY:
        cv2.destroyAllWindows()
    out_csv = os.path.join(OUTPUT_DIR, f"{os.path.splitext(fname)[0]}_timestamps.csv")
    with open(out_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['start_time_sec', 'end_time_sec'])
        for s, e in timestamps:
            writer.writerow([f"{s:.3f}", f"{e:.3f}"])

