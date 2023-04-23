import telebot
import numpy as np
import cv2
import requests
import urllib.request
from io import BytesIO

# инициализация бота
bot = telebot.TeleBot("6274601191:AAF9nDrk2guA-HsqdhGg2qG0U_CLMVMbe1A")
BOT_KEY = "6274601191:AAF9nDrk2guA-HsqdhGg2qG0U_CLMVMbe1A"

# загрузка списка классов
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# загрузка модели YOLOv3-608
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')


# обработка фото с помощью модели
def object_detection(message):
    # скачиваем фото
    file_id = message.photo[-1].file_id
    file_info = bot.get_file(file_id)
    img_url = f"https://api.telegram.org/file/bot{BOT_KEY}/{file_info.file_path}"
    response = requests.get(img_url)
    img_np = np.array(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    # определение размеров фото
    (height, width) = img.shape[:2]

    # создание 4D blob из фото
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (320, 320), swapRB=True, crop=False)

    # загрузка модели и выполнение прямого прохода
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    layer_outputs = net.forward(output_layers)

    # идентификация объектов
    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                # преобразование координаты области детекции
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # добавление объектов и соответствующих данных в списки
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # применение non-maximum suppression для удаления накладывающихся объектов
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # определение количества объектов на фото
    number_detected = len(boxes)
    
    # обозначение объектов на фото
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    font = cv2.FONT_HERSHEY_SIMPLEX

    if len(indices) > 0:
      for i in indices.flatten():
        label = f'{classes[class_ids[i]]}: {round(confidences[i], 2)}'
        box = boxes[i]
        color = colors[class_ids[i]]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        # рисование рамок вокруг объектов
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y - 5), font, 0.5, color, 2)


    # отправка фото с обозначенными объектами
    cv2.imwrite('output.jpg', img)
    with open('output.jpg', 'rb') as f:
        bot.send_photo(message.chat.id, f)

# настройка приветственного сообщения
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.send_message(message.chat.id, """Привет! Отправь мне фото, а я определю объекты на нём. Ещё хотел добавить не пиши команду "/end" иначе будет опасно!  """)

# шутка
@bot.message_handler(commands=['end'])
def send_welcome(message):
    bot.send_message(message.chat.id, """https://www.youtube.com/watch?v=xm3YgoEiEDc""")

# обработка приёма фото
@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    object_detection(message)

# запуск бота
bot.polling()