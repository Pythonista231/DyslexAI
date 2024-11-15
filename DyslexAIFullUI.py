
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.dialogs import Messagebox
from tkinter import filedialog
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import pandas as pd
import numpy as pd
import os
from snapml import LogisticRegression
from openai import OpenAI
import base64
from PIL import Image, ImageTk
import tkinter as tk


openaiApiKey = 'enter your openai api key here'
client = OpenAI(api_key=openaiApiKey)


def testModel(logRegModel): # this is a function that takes a logistic regression model as input and uses testing data to test its accuracy and performance. 
    dfTest = pd.read_csv("C:/Users/micha/Downloads/testingDataCSV.txt") #loading the data from the csv file.   
    xTest = dfTest.drop(columns=['imagePath', 'dyslexic']).values  # features
    yTest = dfTest['dyslexic'].values  

    yPredicted = logRegModel.predict(xTest)
    accuracy = accuracy_score(yTest, yPredicted)
    precision = precision_score(yTest, yPredicted)
    recall = recall_score(yTest, yPredicted)
    f1 = f1_score(yTest, yPredicted)

    print("Model Evaluation:")
    print(f"Accuracy: {accuracy*100:.4f} (% of correct classifications)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")


# function to load the data and train the model, then returns the LogReg model itself. 
def loadModel():
    global df
    df = pd.read_csv("C:/Users/micha/Downloads/trainingDataCSV (1).txt") #loads the data from csv file. 
    x = df.drop(columns=['imagePath', 'dyslexic']).values  # features
    y = df['dyslexic'].values 

    logRegModel = LogisticRegression()
    logRegModel.fit(x, y)
    testModel(logRegModel) # this is a function that tests the model we made for performance and prints the results. 
    return logRegModel


logRegModel = loadModel()



# this is a function which takes an image path and labels the values of its feature, this is so that it could then be used in logistic regression once labelled. 
def labelImage(imagePath):
    global XInputs

    def encodeImage(imagePath): 
        with open(imagePath, "rb") as imageFile: 
            return base64.b64encode(imageFile.read()).decode('utf-8')

    base64Image = encodeImage(imagePath)

    idx = imagePath.rfind('.')
    imageType = imagePath[idx + 1: ]

    try: #openai api call to label the values of the first 3 features. this has to be done in order for the logistic regression model to be used. 
        response = client.chat.completions.create(

            model = 'gpt-4o', 
            messages = [
                {
                    "role": "system", 
                    "content": "You must answer with 4 numbers. like this:a,b,c,d but just with numbers. no spaces at all, just 4 numbers separated by commas."
                },
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text", 
                            "text": "look at the picture, the first you give me number represents the number of corrections a person made, whether it's a scribble or a linethrough etc, any type of correction in their writing counts, but make sure it's not a line from their notebook instead of a correction! if a correction contains more than one word, then treat it as n corrections where n is the number of words corrected, instead of treating it as one large correction. the second number you give me is the number of words that were written overall. third is the number of words that were misspelled. fourth is the number of case mistakes (wherein a person used a capital letter where they shouldn't or the other way around)"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{imageType};base64,{base64Image}"
                            }
                        }
                    ]
                }
            ]

        )
    except Exception as e:
        Messagebox.show_error(f"Error with openai api call: {e}")
        return

    listResponse1 = response.choices[0].message.content.strip().split(',')

    numberOfCorrections = int(listResponse1[0])
    numberOfWords = int(listResponse1[1])
    numberOfMisspelled = int(listResponse1[2])
    numberOfCaseMistakes = int(listResponse1[3])

    # calculate percentages
    percentCorrections = numberOfCorrections / numberOfWords * 100 
    percentMisspelled = numberOfMisspelled / numberOfWords * 100 
    percentCaseMistakes = numberOfCaseMistakes / numberOfWords * 100 

    listResponse1 = [percentCorrections, percentMisspelled, percentCaseMistakes]

    #second api call to label last three features. 
    try:
        response2 = client.chat.completions.create(

            model = 'gpt-4o', 
            messages = [
                {
                    "role": "system", 
                    "content": "you must answer with 3 numbers like this: x,y,z no spaces, just three numbers separated by commas. "
                },
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text", 
                            "text": "look at the picture, I need you to firstly tell me a number between 1 and 3. 1 means that there is no or very little joining between letters, 2 means somewhat and 3 means good joining between letters. second i want you to tell me the legibility score of the handwriting, 1-3 again, 1 means not ledgible, 2 means ledgible, 3 means very ledgible. and lastly i want you to give me a number between 1 to 3, 3 means perfect or near perfect horizontal alignment of handwriting, 2 means medium alignment of handwriting, and 1 means horrible alignment of handwriting, e.g. if it's super sloped. "
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpg;base64,{base64Image}"
                            }
                        }
                    ]
                }
            ]

        
        )
        
    except Exception as e:
        Messagebox.show_error(f"Error with openai api call: {e}")
        return

    listResponse2 = response2.choices[0].message.content.strip().split(',')

    XInputs = listResponse1 + listResponse2
    return XInputs






class Application(ttk.Window):
    def __init__(self):
        super().__init__()

        self.title("Dyslexia Detection Application")
        self.geometry("1500x1000")


        style = ttk.Style()
        style.theme_use('superhero')

        #initialise widgets: 
        self.createWidgets()

    def createWidgets(self): 
        global frames

        # sidebar
        self.sidebar = ttk.Frame(self, bootstyle="primary", width=200)
        self.sidebar.pack(side=LEFT, fill=BOTH)

        #  title
        self.logoLabel = ttk.Label(self.sidebar, text="DyslexAI", font=("Helvetica", 20, "bold"), bootstyle="inverse-primary")
        self.logoLabel.pack(pady=20)

        # Navigation buttons
        self.navButtons = {}
        buttons = [("Home", self.showHome),
                   ("Upload", self.showUpload),
                   ("Analysis", self.showAnalysis)]

        for (text, command) in buttons:
            btn = ttk.Button(self.sidebar, text=text, command=command, bootstyle="outline-light")
            btn.pack(pady=5, fill=X, padx=10)
            self.navButtons[text] = btn
        
        self.container = ttk.Frame(self)
        self.container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        #frames: 
        frames = {}
        for F in (HomePage, UploadPage, AnalysisPage):
            pageName = F.__name__
            frame = F(parent=self.container, controller=self)
            frames[pageName] = frame
            frame.grid(row=0, column=0, sticky=NSEW)
            if F == AnalysisPage: 
                frame.createWidgets()

        

        self.showFrame("HomePage") #default 

    def showFrame(self, pageName):
        frame = frames[pageName]
        frame.tkraise()

    def showHome(self):
        self.showFrame("HomePage")

    def showUpload(self):
        self.showFrame("UploadPage")

    def showAnalysis(self):
        self.showFrame("AnalysisPage")

    def onClosing(self):
        self.destroy()



class HomePage(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(padding=20)
        self.createWidgets()

    def createWidgets(self):
        #title
        titleLabel = ttk.Label(self, text="Welcome to DyslexAI", font=("Helvetica", 24, "bold"), anchor="center", justify="center")
        titleLabel.pack(pady=20, fill="x", expand=True)

        #description
        desc = "Empowering educators and individuals to detect Dyslexia early through Machine Learning."
        titleLabel = ttk.Label(self, text=desc, font=("Helvetica", 14), anchor="center", justify="center")
        titleLabel.pack(pady=10, fill="x", expand=True)

class UploadPage(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.img = None
        self.configure(padding=20)
        self.createWidgets()

    def createWidgets(self):

        # upload area
        titleLabel = ttk.Label(self, text="Upload A Handwriting Sample (~ 3 sentences)", font=("Helvetica", 18, "bold"), anchor="center", justify="center")
        titleLabel.pack(pady=40, fill="x", expand=True)

        # select image button
        selectBtn = ttk.Button(self, text="Select Handwriting Sample Image", command=self.selectImage, bootstyle="success")
        selectBtn.pack(pady=40)

        # File Name Display
        self.fileNameVar = tk.StringVar(value="No file selected")
        fileNameLabel = ttk.Label(self, textvariable=self.fileNameVar, font=("Helvetica", 12))
        fileNameLabel.pack(pady=5)

        # image preview
        self.imagePreview = ttk.Label(self, bootstyle="secondary", anchor="center")
        self.imagePreview.pack(pady=10, padx=50, fill=BOTH, expand=True)

        # analysis messages (2): 
        self.analysisMessageVar = tk.StringVar(value="")
        analysisMsgLabel = ttk.Label(self, textvariable=self.analysisMessageVar, font=("Helvetica", 12), wraplength=800, justify="center")
        analysisMsgLabel.pack(pady=5)

        self.analysisMessageVar2 = tk.StringVar(value="")
        analysisMsgLabel2 = ttk.Label(self, textvariable=self.analysisMessageVar2, font=("Helvetica", 12), wraplength=800, justify="center")
        analysisMsgLabel2.pack(pady=5)

    def selectImage(self):
        global imagePath
        imagePath = filedialog.askopenfilename(title="Select an Image File",
                                               filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;")])
        if imagePath:
            self.displayAndLabelImage(imagePath)

    def displayAndLabelImage(self, imagePath):
        global XInputs
        if imagePath.lower().endswith(('.png', '.jpg', '.jpeg')):
            self.imagePath = imagePath
            self.imageType = imagePath.split('.')[-1]
            fileName = os.path.basename(imagePath)
            self.fileNameVar.set(f"Selected File: {fileName}")

            # Load and display image
            try:
                img = Image.open(imagePath)
                img.thumbnail((400, 400))
                self.img = ImageTk.PhotoImage(img)
                self.imagePreview.configure(image=self.img, borderwidth=2, relief="groove")
            except Exception as e:
                Messagebox.show_error("Invalid image", "The selected file is not a valid image.")
                self.imagePreview.configure(image='')
                self.fileNameVar.set("No file selected")
                self.analysisMessageVar.set("")
                self.analysisMessageVar2.set("")
                return

            self.analysisMessageVar.set("Valid image, click 'Analysis' to start Machine Learning inference")
            self.analysisMessageVar2.set("")

            XInputs = labelImage(imagePath)
            


        else:
            # the openai api only accepts a few image file types. this is to accomodate for that. 
            Messagebox.show_warning("Invalid File", "Please select a valid image file.")

class AnalysisPage(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.analysisResults = None
        self.currentFactor = '% corrections to words'  # Default factor
        self.configure(padding=20)

    def createWidgets(self):
        mainLayout = ttk.Frame(self)
        mainLayout.pack(fill=BOTH, expand=True, padx=10, pady=10)

        # Title
        titleLabel = ttk.Label(mainLayout, text="Image Analysis", font=("Helvetica", 20, "bold"))
        titleLabel.pack(pady=10)

        # Start analysis button
        self.startAnalysisBtn = ttk.Button(mainLayout, text="Start Analysis", command=self.startAnalysis, bootstyle="success")
        self.startAnalysisBtn.pack(pady=10)

        # Progress bar 
        self.progress = ttk.Progressbar(mainLayout, mode='determinate', length=400)
        self.progress.pack(pady=10)

        # Analysis results sections
        resultsFrame = ttk.Frame(mainLayout)
        resultsFrame.pack(pady=10, fill=BOTH, expand=True)

        # Left Side: graph and factor buttons 
        leftFrame = ttk.Frame(resultsFrame)
        leftFrame.pack(side=LEFT, fill=BOTH, expand=True, padx=10)
        
        # features. 
        factors = ['% corrections to words', '% spelling error', '% case mistakes', 'joining', 'legibility', 'line alignment']
        self.factorVar = tk.StringVar(value=self.currentFactor)

        buttonFrame = ttk.LabelFrame(leftFrame, text="Select Factor")
        buttonFrame.pack(pady=5)

        for factor in factors:
            btn = ttk.Radiobutton(buttonFrame, text=factor, variable=self.factorVar, value=factor, command=self.changeFactor, bootstyle="info")
            btn.pack(side=LEFT, padx=2)

        # Graph title
        self.graphTitle = ttk.Label(leftFrame, text=self.currentFactor, font=("Helvetica", 14, "bold"))
        self.graphTitle.pack(pady=5)

        #matplotlib figure. 
        self.figure = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)  # Create an Axes object
        self.canvas = FigureCanvasTkAgg(self.figure, master=leftFrame)
        self.canvas.get_tk_widget().pack(fill = BOTH, expand=  True)       

        
        # Right side : results
        rightFrame = ttk.Frame(resultsFrame)
        rightFrame.pack(side=RIGHT, fill=BOTH, expand=True, padx=10)

        # Percentages and results frame
        resultsInnerFrame = ttk.LabelFrame(rightFrame, text="Analysis Results")
        resultsInnerFrame.pack(fill=BOTH, expand=True)

        # Percentages labels
        for factor in factors:
            frame = ttk.Frame(resultsInnerFrame)
            frame.pack(anchor="w", padx=10, pady=2)
            if factor.strip() == 'line alignment' or factor.strip() == 'legibility' or factor.strip() == 'joining': 
                lbl = ttk.Label(frame, text=f"{factor} (1-3):", font=("Helvetica", 12, "bold"))
            else: 
                lbl = ttk.Label(frame, text=f"{factor}:", font=("Helvetica", 12, "bold"))
            lbl.pack(side=LEFT)
            val = ttk.Label(frame, text="N/A", font=("Helvetica", 12))
            val.pack(side=LEFT, padx=5)
            frames["AnalysisPage"].resultLabels = getattr(self, 'resultLabels', {})
            self.resultLabels[factor] = val

        # Dyslexia prob: 
        dysLabel = ttk.Label(resultsInnerFrame, text="% Chance of Dyslexia:", font=("Helvetica", 16, "bold"))
        dysLabel.pack(pady=25)
        self.dyslexiaVar = tk.StringVar(value="N/A")
        dysValue = ttk.Label(resultsInnerFrame, textvariable=self.dyslexiaVar, font=("Helvetica", 12))
        dysValue.pack()

    def startAnalysis(self): # this is run by the startAnalysis button. 
        uploadPage = frames["UploadPage"]
        if not imagePath:
            Messagebox.show_warning("No Image", "Please upload an image in the Upload section.")
            return

        self.startAnalysisBtn.state(['disabled'])
        self.progress['value'] = 0
        self.runAnalysis()

    def runAnalysis(self):
        global XInputs, XInputsNumpy
        # Simulate analysis time
        for i in range(1, 101):
            time.sleep(0.01)
            self.progress['value'] = i
            self.update_idletasks()

        # Perform analysis

        #finding their probability of dyslexia: 
        
        XValsDict = {
            r'% corrections to words': float(XInputs[0]), 
            r'% spelling error':float(XInputs[1]), 
            r'% case mistakes':float(XInputs[2]), 
            r'joining':float(XInputs[3]),
            r'legibility':float(XInputs[4]),
            r'line alignment':float(XInputs[5])
              
             }
        XDf = pd.DataFrame(XValsDict, index = [0])
        XInputsNumpy = XDf.to_numpy()
        XInputsNumpy = XInputsNumpy.reshape(1, -1)

        dyslexiaProb = float(logRegModel.predict_proba(XInputsNumpy)[:,1][0]) * 100

        
        # Prepare results
        yourResults = {}
        factors = [r'% corrections to words', r'% spelling error', r'% case mistakes', 'joining', 'legibility', 'line alignment']
        for idx, factor in enumerate(factors):
            yourResults[factor] = XInputs[idx]

        self.analysisResults = {
            'yourResults': yourResults
        }

        # Update GUI with results
        self.after(0, self.updateResults, dyslexiaProb)

    def updateResults(self, dyslexiaProbability):
        # Update "Your Results" labels
        for factor, value in self.analysisResults['yourResults'].items():
            self.resultLabels[factor].config(text=f"{round(float(value), 3)}")

        self.dyslexiaVar.set(f"{round(float(dyslexiaProbability), 3)}%")

        # Display graph for the current factor
        self.showGraph(self.currentFactor)

        self.startAnalysisBtn.state(['!disabled'])

    def changeFactor(self):
        self.currentFactor = self.factorVar.get()
        self.graphTitle.config(text=self.currentFactor)
        self.showGraph(self.currentFactor)

    def showGraph(self, factor):
        if not hasattr(self, 'analysisResults'):
            Messagebox.show_warning("No Results", 'Please perform analysis first - Click "Start Analysis".')
            return

        yourResult = float(self.analysisResults['yourResults'].get(factor, 0.0))

        # Clear the previous plot
        self.ax.clear()

        # Plot training datapoints
        dyslexic = df['dyslexic']
        YValues = df[factor]
        colors = ['green' if x == 0 else 'red' for x in dyslexic]
        self.ax.scatter(range(1, len(YValues)+1), YValues, c=colors, label='Training datapoints (red=dyslexic, green=non-dyslexic)')

        # Plot "Your Result" line
        self.ax.axhline(y=yourResult, color='blue', linestyle='-', label='Your Result')

        self.ax.set_xlabel("Training Datapoint Number")
        self.ax.set_ylabel(f"{factor} (%)")
        self.ax.set_title(factor.capitalize())
        self.ax.legend()

        self.figure.tight_layout()
        self.canvas.draw()






app = Application()
app.mainloop()

