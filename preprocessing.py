import os
import cv2
import csv

# Configuration parameters
VIDEO_DIR = './data/shot_vids'
OUTPUT_DIR = './data/shot_timestamps'
ROI_RATIO = (0.5, 0.5)  # top-left quadrant
MIN_CONSECUTIVE_NO_DETECT = 10  # frames to confirm end of shot
HOUGH_DP = 1.2
HOUGH_MIN_DIST = 20
HOUGH_PARAM1 = 50
HOUGH_PARAM2 = 30
HOUGH_MIN_RADIUS = 5
HOUGH_MAX_RADIUS = 50
# Only accept new detections when circle center x is within this fraction of ROI width
NEW_SHOT_ENTRY_X_FRAC = 0.3
DISPLAY = True  # show live footage with overlays

os.makedirs(OUTPUT_DIR, exist_ok=True)

def detect_ball(cropped_gray, fgmask):
    """
    Use HoughCircles on the masked, blurred ROI to detect a circular ball.
    Returns circle parameters (x, y, r) or None, applying size and motion filters.
    """
    # Mask out static background: only moving pixels
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
    if not fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        continue

    video_path = os.path.join(VIDEO_DIR, fname)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define ROI boundaries
    roi_w = int(width * ROI_RATIO[0])
    roi_h = int(height * ROI_RATIO[1])

    # Background subtractor to focus on moving objects
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

        # Crop to top-left ROI
        roi = frame[0:roi_h, 0:roi_w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Background subtraction + morphological opening to remove noise
        fgmask = subtractor.apply(roi)
        _, fgmask = cv2.threshold(fgmask, 254, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Detect ball only on moving regions
        circle = detect_ball(blurred, fgmask)
        ball_present = False
        if circle is not None:
            x, y, r = circle
            # Only treat as new entry if within left region of ROI (filters static logos)
            if not in_shot:
                if x < roi_w * NEW_SHOT_ENTRY_X_FRAC:
                    ball_present = True
            else:
                # If already tracking a shot, accept any motion-based detection
                ball_present = True

        # Overlay for visualization
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (roi_w, roi_h), (255, 0, 0), 2)
        if ball_present:
            cv2.circle(overlay, (x, y), r, (0, 255, 0), 2)

        # Shot start/end logic
        if ball_present and not in_shot:
            in_shot = True
            start_frame = frame_idx
            no_detect_count = 0
            cv2.putText(overlay, 'SHOT START', (50, height - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif in_shot and not ball_present:
            no_detect_count += 1
            if no_detect_count >= MIN_CONSECUTIVE_NO_DETECT:
                end_frame = frame_idx - no_detect_count
                timestamps.append((start_frame / fps, end_frame / fps))
                in_shot = False
                start_frame = None
                no_detect_count = 0
                cv2.putText(overlay, 'SHOT END', (50, height - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display timestamp
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

    # Write timestamps to CSV
    out_csv = os.path.join(OUTPUT_DIR, f"{os.path.splitext(fname)[0]}_timestamps.csv")
    with open(out_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['start_time_sec', 'end_time_sec'])
        for s, e in timestamps:
            writer.writerow([f"{s:.3f}", f"{e:.3f}"])

    print(f"Processed {fname}, found {len(timestamps)} shots, saved to {out_csv}")

print("Done processing all videos.")
