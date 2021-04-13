# импорт библиотеки распознавания
import cv2

############################
# задание размер окна отображения и цвета
frameWidth = 640;
frameHeight = 480;
minArea = 500;
color = (255, 0, 255);

# подключение натренерованной модели (метод каскада Хаара) на российские номера автомобилей

nPlateCascade = cv2.CascadeClassifier("Resources/haarcascade_russian_plate_number.xml")

#################################
# подключение видеокамеры и установка её значений
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)
count = 0;
# процесс сравнения изображения в видеопотоке с моделью, а также сохранение
while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# обнаружение номера и задание размера отображения в другом окне
    numberPlates = nPlateCascade.detectMultiScale(imgGray, 1.1, 4)

    for (x, y, w, h) in numberPlates:
        area = w * h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(img, "Number", (x, y - 5),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
            imgRoi = img[y:y + h, x:x + w]
            cv2.imshow("ROI", imgRoi)

    cv2.imshow("Result", img)
# процесс сохранения отобранного номера
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("Resources/Scanned/NoPlate_" + str(count) + ".jpg", imgRoi)
        cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Saved", (150, 265), cv2.FONT_HERSHEY_DUPLEX,
                    2, (0, 0, 255), 2)
        cv2.imshow("Result", img)
        cv2.waitKey(500)
        count += 1
