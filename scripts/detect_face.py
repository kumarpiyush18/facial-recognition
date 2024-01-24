
import cv2

harrcascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


if __name__ == '__main__':
    img = cv2.imread("output/7eedaf5b4b.jpg")
    faces = harrcascade.detectMultiScale(img,1.3,2)
    print(faces)
