import cv2
import easyocr

reader = easyocr.Reader(['en'])
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 100, 200)

    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    plate = None

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(contour)
            plate = frame[y:y+h, x:x+w]
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            break

    if plate is not None:
        result = reader.readtext(plate)
        for (bbox, text, prob) in result:
            cv2.putText(frame, text, (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0,255,0), 2)

    cv2.imshow("License Plate Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
