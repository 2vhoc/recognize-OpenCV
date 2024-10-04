import tkinter as tk
from tkinter import filedialog
import cv2, os
import numpy as np
from PIL import Image, ImageTk

main = tk.Tk()
pathIMG = []
cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')
count = 0
photo = None
dx = None
dy = None
dw = None
dh = None


def take1():
    global count, photo, dx, dy, dw, dh
    print(dx, dy, dw, dh, "Hết")
    if photo is not None:
        count += 1
        gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f'datasets/User.{count}.jpg', gray[dy:dy + dh, dx:dx + dh])  # Lưu ảnh đã chụp
        print(f'Số ảnh được chụp là {count} và đã lưu tại datasets/User.{count}.jpg')
def selectIMG():
    file = filedialog.askopenfilename(title='Select Image', filetypes=[("Image File", "*.jpg;*.jpeg;*.png;*.gif")])
    pathIMG.append(file)
    print(pathIMG)
    global detector
    global count, photo, dx, dy, dw, dh
    for img in pathIMG:
        img = cv2.imread(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            dx = x
            dy = y
            dw = w
            dh = h
            photo = img
            take1()

def selectImgByVideo():
    def updateFrame():
        global photo, dx, dy, dw, dh
        ret, frame = cam.read()
        if not ret:
            print("Không thể đọc khung hình từ camera.")
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        photo = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        imgTK = ImageTk.PhotoImage(Image.fromarray(frame))
        canvas.create_image(55, 100, anchor=tk.NW, image=imgTK)
        canvas.imgtk = imgTK
        for x, y, w, h in faces:
            dx = x
            dy = y
            dw = w
            dh = h

        main.after(10, updateFrame)

    tk.Button(main, text='Take A Photo', command=take1).grid(row=10, column=5, sticky=tk.W, padx=5, pady=5)
    updateFrame()
def getImgLabel(path):
    imgPaths = [os.path.join(path, x) for x in os.listdir(path)]
    faceSamples = []
    ids = []
    for imgPath in imgPaths:
        pilImg = Image.open(imgPath).convert('L')
        imgNp = np.array(pilImg, 'uint8')
        id = int(os.path.split(imgPath)[-1].split('.')[1])
        faces = detector.detectMultiScale(imgNp)
        for (x, y, w, h) in faces:
            faceSamples.append(imgNp[y:y + h, x:x + w])
            ids.append(id)
        return faceSamples, ids
def train():
    path = 'datasets'
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, ids =  getImgLabel(path)
    recognizer.train(faces, np.array(ids))
    recognizer.write('datatrain/dataTrain.yml')
    print('Train Thành Công')
def Recognizer():
    print('ok')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('datatrain/dataTrain.yml')
    faceCascade = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')
    font = cv2.FONT_HERSHEY_SIMPLEX
    id = 0
    Names = ['Vu Hoc', 'Hoc 1', 'Hoc 2', 'Hoc 3', 'Hoc 4', 'Hoc 5', 'Hoc 6']
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(3, 480)
    minW = 0.1 * cap.get(4)
    minH = 0.1 * cap.get(3)
    def update():
        global ok1
        ret, frame = cam.read()
        if not ret:
            print("Không thể đọc khung hình từ camera.")
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            id, confident = recognizer.predict(gray[y:y + h, x:x + w])
            id = Names[id]
            if confident < 100:
                confident = f'   {(round(100 - confident))}%'
                cv2.putText(frame, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
                cv2.putText(frame, str(confident), (x + 5, y - h + 5), font, 1, (0, 255, 0), 2)
            imgTK = ImageTk.PhotoImage(Image.fromarray(frame))
            canvas.create_image(55, 100, anchor=tk.NW, image=imgTK)
            canvas.imgtk = imgTK

        main.after(10, update)

    update()
def struct():
    global canvas
    canvas = tk.Canvas(main, width=700, height=700)
    canvas.grid(row=0, column=2)
    x1, y1 = 50, 50
    x2, y2 = 700, 700
    canvas.create_rectangle(x1, y1, x2, y2, outline="black", fill="gray")
    tk.Button(main, text='Recognizer', command=Recognizer).grid(row=10, column=5)
    tk.Button(main, text='Select Image By Video', command=selectImgByVideo).grid(row=10, column=3)
    tk.Button(main, text='Train', command=train).grid(row=10, column=4)
    tk.Button(main, text='Select Image', command=selectIMG).grid(row=10, column=2)

if __name__ == '__main__':
    struct()
    main.geometry('1200x800')
    main.mainloop()
    cam.release()
    cv2.destroyAllWindows()
