from tkinter import *
import tkinter
# pip install pillow
from PIL import Image, ImageTk
import PIL
import json
from MyVideoCapture import MyVideoCapture
from MyDetector import MyDetector
import cv2


class App:
    ConfigJsonFile = {}
    NumZones = 0
    pts = []
    DrawZones = False
    DoDetect = False
    stop = False
    xinit = 0
    yinit = 0
    xend = 0
    yend = 0
    LastFrame = []

    def __init__(self, window, window_title):
        self.ConfigJsonFile["Zones"] = []
        self.window = window
        self.window.title(window_title)
        self.video_source = "http://199.48.198.27/mjpg/video.mjpg"
        # open video source
        self.vid = MyVideoCapture(self.video_source)
        # Load Detector
        self.LABELS_FILE = 'coco_labels.txt'
        self.WEIGHTS_FILE = 'mask_rcnn_coco.h5'

        self.detector = MyDetector(self.LABELS_FILE, self.WEIGHTS_FILE)

        self.ZoneHistoryFrame = tkinter.Frame(window)
        self.ZoneHistoryFrame.grid(row=0, column=1)

        self.lbl = Label(self.ZoneHistoryFrame,
                         text="A list of Zones Drawers...")

        self.listbox = Listbox(self.ZoneHistoryFrame, selectmode='multiple')

        self.DeleteZone_Button = tkinter.Button(
            self.ZoneHistoryFrame, text="delete", command=self.DeleteZone_Button_Callback)

        self.CommandFrame = tkinter.Frame(window)
        self.CommandFrame.grid(row=0, column=0)

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(
            self.CommandFrame, width=self.vid.width, height=self.vid.height)
        self.canvas.bind("<B1-Motion>", self.Moving)
        self.canvas.bind("<ButtonPress-1>", self.Press)
        self.canvas.bind("<ButtonPress-3>", self.SaveZone)

        self.DrawZones_Button = tkinter.Button(
            self.CommandFrame, text="Draw zones", command=self.DrawZones_Button_Callback)

        self.Detect_Button = tkinter.Button(
            self.CommandFrame, text="Detect", command=self.Detect_Button_Callback)

        self.canvas.grid(row=0, column=0)
        self.DrawZones_Button.grid(row=1, column=0)
        self.DeleteZone_Button.grid(row=2, column=0)

        self.lbl.grid(row=0, column=0)
        self.Detect_Button.grid(row=3, column=0)
        self.listbox.grid(row=1, column=0)

 # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()
        self.ActualizeZonesList()
        self.window.mainloop()

    def DeleteZone_Button_Callback(self):
        print("sad")
        Selections = self.listbox.curselection()
        auxConfigJsonFile = self.ConfigJsonFile.copy()
        self.ConfigJsonFile["Zones"] = []
        print(len(auxConfigJsonFile["Zones"]))
        for i in range(len(auxConfigJsonFile["Zones"])):
            print("CARENALGA")

            self.listbox.delete(i)
            if not i in Selections[::-1]:
                print(auxConfigJsonFile["Zones"][i])
                self.ConfigJsonFile["Zones"].append(
                    auxConfigJsonFile["Zones"][i])

        with open("Config.json", 'w') as f:
            json.dump(self.ConfigJsonFile, f)
        self.ActualizeZonesList()

    def ActualizeZonesList(self):
        # ZONES
        try:
            with open("Config.json", 'r') as f:
                self.ConfigJsonFile = json.load(f)

            for i in range(0, len(self.ConfigJsonFile["Zones"])):

                self.listbox.delete(i)
                print("ZONE "+str(i)+": ("+str(self.ConfigJsonFile["Zones"][i]["xinit"])+"," + str(self.ConfigJsonFile["Zones"][i]["yinit"])+"),("+str(
                    self.ConfigJsonFile["Zones"][i]["xend"])+","+str(self.ConfigJsonFile["Zones"][i]["yend"])+")")
                self.listbox.insert(i,
                                    "ZONE "+str(i)+": ("+str(self.ConfigJsonFile["Zones"][i]["xinit"])+"," + str(self.ConfigJsonFile["Zones"][i]["yinit"])+"),("+str(self.ConfigJsonFile["Zones"][i]["xend"])+","+str(self.ConfigJsonFile["Zones"][i]["yend"])+")")

            print("GOOD JSON")
        except ValueError as e:
            print("BAAAD JSON, THERE IS NO ZONES!")

    def DrawZones_Button_Callback(self):

        if self.DrawZones_Button.config('text')[-1] == 'Draw Zones':
            self.stop = True
            self.DrawZones = True
            self.DrawZones_Button.config(text='Stop Drawing')
            if self.DoDetect:
                self.DoDetect = False
                self.Detect_Button.config(text='Detect')
        else:
            self.stop = False
            self.DrawZones = False
            self.DrawZones_Button.config(text='Draw Zones')

    def Detect_Button_Callback(self):

        if self.Detect_Button.config('text')[-1] == 'Detect':
            self.Detect_Button.config(text='Stop detection')
            self.DoDetect = True

            if self.DrawZones:
                self.stop = False
                self.DrawZones = False
                self.DrawZones_Button.config(text='Draw Zones')
        else:
            self.DoDetect = False
            self.Detect_Button.config(text='Detect')

    def SaveZone(self, event):
        if(self.xinit > self.xend):
            aux = self.xinit
            self.xinit = self.xend
            self.xend = aux
        if(self.yinit > self.yend):
            aux = self.yinit
            self.yinit = self.yend
            self.yend = aux
        print("INIT POINT", self.xinit, self.yinit)
        print("END POINT", self.xend, self.yend)
        print("Saving Zone")
        AuxJson = {}
        self.NumZones += 1
        AuxJson["id"] = self.NumZones
        AuxJson["xinit"] = self.xinit
        AuxJson["yinit"] = self.yinit
        AuxJson["xend"] = self.xend
        AuxJson["yend"] = self.yend
        self.ConfigJsonFile["Zones"].append(AuxJson)
        print(self.ConfigJsonFile)
        with open("Config.json", 'w') as f:
            json.dump(self.ConfigJsonFile, f)
        self.ActualizeZonesList()

    def Moving(self, event):

        self.xend = event.x
        self.yend = event.y
        print("Released at", event.x, event.y)

    def Press(self, event):

        self.xinit = event.x
        self.yinit = event.y
        print("Pressed at", event.x, event.y)

    def update(self):

        if self.DrawZones:
            try:
                for i in range(0, len(self.ConfigJsonFile["Zones"])):
                    cv2.rectangle(self.LastFrame,
                                  (self.ConfigJsonFile["Zones"][i]["xinit"], self.ConfigJsonFile["Zones"][i]["yinit"]), (self.ConfigJsonFile["Zones"][i]["xend"], self.ConfigJsonFile["Zones"][i]["yend"]), (255, 255, 0), 2)
                    cv2.putText(self.LastFrame, "ZONE "+str(i),
                                (self.ConfigJsonFile["Zones"][i]["xinit"], self.ConfigJsonFile["Zones"][i]["yinit"]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), lineType=cv2.LINE_AA, thickness=2)
             #   print("GOOD JSON")
            except ValueError as e:
                print("BAAAD JSON, THERE IS NO ZONES!")

            self.photo = PIL.ImageTk.PhotoImage(
                image=PIL.Image.fromarray(cv2.rectangle(self.LastFrame.copy(), (self.xinit, self.yinit),
                                                        (self.xend, self.yend), (0, 255, 0),	1)))

            self.canvas.create_image(
                0, 0, image=self.photo, anchor=tkinter.NW)

        if not self.stop:
            ret, frame = self.vid.get_frame()

            if ret:
                self.LastFrame = frame

                if self.DoDetect:
                    frame = self.detector.Detect(frame.copy())

                    frame=self.detector.WatchIfCarOnZone(frame, self.ConfigJsonFile)
                    print("BITCH")

                try:
                    for i in range(0, len(self.ConfigJsonFile["Zones"])):
                        cv2.rectangle(frame,
                                      (self.ConfigJsonFile["Zones"][i]["xinit"], self.ConfigJsonFile["Zones"][i]["yinit"]), (self.ConfigJsonFile["Zones"][i]["xend"], self.ConfigJsonFile["Zones"][i]["yend"]), (255, 255, 0), 2)
                        cv2.putText(frame, "ZONE "+str(i),
                                    (self.ConfigJsonFile["Zones"][i]["xinit"], self.ConfigJsonFile["Zones"][i]["yinit"]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), lineType=cv2.LINE_AA, thickness=2)
                #    print("GOOD JSON")
                except ValueError as e:
                    print("BAAAD JSON, THERE IS NO ZONES!")

                self.photo = PIL.ImageTk.PhotoImage(
                    image=PIL.Image.fromarray(frame))
                self.canvas.create_image(
                    0, 0, image=self.photo, anchor=tkinter.NW)
        self.window.after(self.delay, self.update)

 # Create a window and pass it to the Application object
App(tkinter.Tk(), "Tkinter and OpenCV")
