import cv2, time
video = cv2.VideoCapture(0)
time.sleep(3)
first_frame = None
while True:
    check, frame = video.read()
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_image, (25, 25), 0)

    if first_frame is None:
        first_frame = blur
        continue

    img_diff = cv2.absdiff(first_frame, blur)
    thresh_img = cv2.threshold(img_diff, 40, 255, cv2.THRESH_BINARY)[1]
    thresh_img = cv2.dilate(thresh_img, None, iterations=5)

    (contours, _) = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < 9000:
            continue
        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

    cv2.imshow("captured", gray_image)
    cv2.imshow("diff", img_diff)
    cv2.imshow("thresh_frame", thresh_img)
    cv2.imshow("detected", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
video.release()
cv2.destroyAllWindows

