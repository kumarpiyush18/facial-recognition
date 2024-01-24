import cv2
vid = cv2.VideoCapture(0)

if __name__ == '__main__':
    while True:
        _, frame = vid.read()

        # Display the resulting frame
        cv2.imshow('my_frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()