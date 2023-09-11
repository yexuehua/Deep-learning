import cv2

# Video file path
video_file = 'your_video.mp4'

# Load the video
cap = cv2.VideoCapture(video_file)

# Load the small part of the frame
part_to_match = cv2.imread('part_to_match.png', cv2.IMREAD_GRAYSCALE)

# Initialize ORB detector
orb = cv2.ORB_create()

# Find the keypoints and descriptors in the part to match
keypoints_part, descriptors_part = orb.detectAndCompute(part_to_match, None)

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        print("End of video reached. The part was not found.")
        break

    # Convert the frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the keypoints and descriptors in the frame
    keypoints_frame, descriptors_frame = orb.detectAndCompute(frame_gray, None)

    # Create a Brute Force Matcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match the descriptors
    matches = bf.match(descriptors_part, descriptors_frame)

    # Sort the matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # You can adjust the threshold to control the matching sensitivity
    threshold = 50  # Adjust this threshold as needed

    if len(matches) > threshold:
        print("Part found in frame:", cap.get(cv2.CAP_PROP_POS_FRAMES))
        break

# Release the video capture object
cap.release()
