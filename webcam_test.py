import cv2
import mediapipe as mp
import time

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

camera = cv2.VideoCapture(0)

# Path to where the trained face model is stored
MODEL_PATH = "models/face_landmarker.task"

# Think of this as the settings for the face landmaker, Configuration object
options = FaceLandmarkerOptions(
    # First setting: which model?
    base_options = BaseOptions(
        model_asset_path = MODEL_PATH
    ),
    # Second setting: We are running the model on a video, not an image
    running_mode = RunningMode.VIDEO,
    # Third setting: how many faces to detect?
    num_faces = 1,
    # Fourth setting: what to output? Face blendshapes represent facial expressions and their scores
    output_face_blendshapes = True
)

# Create the face landmarker from the options
landmarker = FaceLandmarker.create_from_options(options)

# Start the timer to measure the time it takes to process the frame
start_time = time.perf_counter()

BROW_ON_THRESHOLD = 0.12
BROW_OFF_THRESHOLD = 0.08

SMILE_ON_THRESHOLD = 0.40
SMILE_OFF_THRESHOLD = 0.15
SMILE_JAW_LIMIT = 0.35

JAW_MIN = 0.05
JAW_MAX = 0.80

distortion_active = False
chorus_active = False
wah_position = 0.0

# Keep the camera open and displaying until otherwise told
while True:
    success, frame = camera.read()

    # If opencv failed to get a frame from the camera, break out of the loop so nothing is displayed
    if not success:
        break

    # OpenCV uses BGR format, but Mediapipe uses RGB format
    # So we convert the frame from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format = mp.ImageFormat.SRGB,
        data = rgb_frame
    )

    # Calculate the timestamp of the frame in milliseconds
    timestamp_ms = int((time.perf_counter() - start_time) * 1000)

    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    # If the face landmarks are detected, get the blendshapes of the first face
    if result.face_blendshapes:
        blendshapes = result.face_blendshapes[0]

        # Create a dictionary to store the blendshape values
        blendshape_values = {}

        for item in blendshapes:
            blendshape_values[item.category_name] = item.score

        jaw_open = blendshape_values.get("jawOpen", 0.0)
        wah_position = (jaw_open - JAW_MIN) / (JAW_MAX - JAW_MIN)
        wah_position = max(0.0, min(1.0, wah_position))
        
        smile_left = blendshape_values.get("mouthSmileLeft", 0.0)
        smile_right = blendshape_values.get("mouthSmileRight", 0.0)

        brow_left = blendshape_values.get("browDownLeft", 0.0)
        brow_right = blendshape_values.get("browDownRight", 0.0)

        brow_down = (brow_left + brow_right) / 2

        smile = (smile_left + smile_right) / 2

        if not distortion_active and brow_down > BROW_ON_THRESHOLD:
            distortion_active = True
        elif distortion_active and brow_down < BROW_OFF_THRESHOLD:
            distortion_active = False

        if not chorus_active and smile > SMILE_ON_THRESHOLD and jaw_open < SMILE_JAW_LIMIT:
            chorus_active = True
        elif chorus_active and (smile < SMILE_OFF_THRESHOLD or jaw_open > SMILE_JAW_LIMIT):
            chorus_active = False

        print(
            "\nJaw Open: ", jaw_open, 
            "\nWah Position: ", wah_position,
            "\nSmile: ", smile,
            "\nBrow Down: ", brow_down,
            "\nDistortion: ", distortion_active,
            "\nChorus: ", chorus_active
            )
    else:
        distortion_active = False
        chorus_active = False
        wah_position = 0.0

    # Display the frame that it read from the camera stored in the frame variable
    cv2.imshow("Face Pedal Camera", frame)

    # Check if user pressed the 'q' key to end the program, if so break out of the loop
    # Effectively asking, if key_pressed == q_key
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up the camera and close the window 
camera.release()
cv2.destroyAllWindows()