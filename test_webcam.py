import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Webcam connected successfully!")
print("Press Q to close the webcam window.")

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read webcam frame.")
        break

    cv2.imshow("ARGUS-X Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("Webcam test completed.")