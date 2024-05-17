
import os

import cv2

from slam import process_frame


def main(file: str) -> None:
    if file is None or file == "":
        # Video stream from webcam
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(file)

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            # Process video frame
            process_frame(frame)
        else:
            break

    print(f"Releasing video capture and destroying all windows")
    cap.release()
    cv2.destroyAllWindows()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    vid = "./test_vids/test_ohio.mp4"
    if not os.path.isfile(vid):
        raise FileNotFoundError

    main(file=vid)

# TODO: 07:22:30

