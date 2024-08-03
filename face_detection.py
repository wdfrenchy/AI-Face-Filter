import cv2
from deepface import DeepFace

overlay_image_path = 'glass.png'
overlay_img = cv2.imread(overlay_image_path, -1) # Load with transparency

def transparentOverlay(src, overlay, pos=(0, 0), scale=1):
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape  # Size of foreground
    rows, cols, _ = src.shape  # Size of background Image
    y, x = pos[0], pos[1]  # Position of foreground/overlay image

    for i in range(h):
        for j in range(w):
            if x + i >= rows or y + j >= cols:
                continue
            alpha = float(overlay[i][j][3] / 255.0)  # read the alpha channel
            src[x + i][y + j] = alpha * overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]
    return src

# Initialize the Haar Cascade face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the Haar Cascade eye detection model, considering glasses
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale (required by Haar Cascades)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Limit eye detection to the upper half of the face
        roi_gray_upper_half = gray[y:y+int(h/2), x:x+w]
        eyes = eye_cascade.detectMultiScale(
            roi_gray_upper_half, scaleFactor=1.1, minNeighbors=5)

        for (ex, ey, ew, eh) in eyes:
            # Basic post-processing check: Eye aspect ratio (to reduce false positives, e.g., detecting a nose as eyes)
            aspect_ratio = ew / eh
            if 0.75 < aspect_ratio < 1.3:
                # Adjusting the y coordinate since we are searching in the upper half
                cv2.rectangle(frame, (x + ex, y + ey),
                              (x + ex + ew, y + ey + eh), (255, 0, 0), 2)
            scale_factor = w / overlay_img.shape[1]

            # Call the transparent overlay function
            frame = transparentOverlay(frame, overlay_img, pos=(x,y+ey), scale = scale_factor)
                

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
