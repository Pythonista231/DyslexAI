
import ttkbootstrap as ttk
# import tkinter as tk
from ttkbootstrap.constants import *
from ttkbootstrap.dialogs import Messagebox
# from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import pandas as pd
import os
from snapml import LogisticRegression
from openai import OpenAI
import base64
from PIL import Image, ImageTk
import tkinter as tk


openai_api_key = '' # enter your api key there. 
client = OpenAI(api_key=openai_api_key)


# Load data and train the model and will return it. 
def load_model():
    global LogRegModel, df
    df = pd.read_csv("C:/Users/micha/Documents/IBMHackathon/images_dataset.csv")
    
    x = df.drop(columns=['image_path', 'dyslexic']).values  # features
    y = df['dyslexic'].values

    LogRegModel = LogisticRegression()
    LogRegModel.fit(x, y)
    return LogRegModel

LogRegModel = load_model()



# this is a function which takes an image path and labels its features.
def label_image(image_path):
    global XInputs

    def encode_image(image_path): 
        with open(image_path, "rb") as image_file: 
            return base64.b64encode(image_file.read()).decode('utf-8')

    base64_image = encode_image(image_path)

    idx = image_path.rfind('.')
    image_type = image_path[idx + 1:]

    try: 
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
                                "url": f"data:image/{image_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]

        )
    except Exception as e:
        Messagebox.show_error(f"Error: {e}")
        return

    list_response1 = response.choices[0].message.content.strip().split(',')

    numberOfCorrections = int(list_response1[0])
    numberOfWords = int(list_response1[1])
    numberOfMisspelled = int(list_response1[2])
    numberOfCaseMistakes = int(list_response1[3])

    # calculate percentages
    percent_corrections = numberOfCorrections / numberOfWords * 100 
    percent_misspelled = numberOfMisspelled / numberOfWords * 100 
    percent_case_mistakes = numberOfCaseMistakes / numberOfWords * 100 

    list_response1 = [percent_corrections, percent_misspelled, percent_case_mistakes]

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
                            "text": "look at the picture, I need you to firstly tell me a number between 1 and 3. 1 means that there is no or very little joining between letters, 2 means somewhat and 3 means good joining between letters. second i want you to tell me the ledgibility score of the handwriting, 1-3 again, 1 means not ledgible, 2 means ledgible, 3 means very ledgible. and lastly i want you to give me a number between 1 to 3, 3 means perfect or near perfect horizontal alignment of handwriting, 2 means medium alignment of handwriting, and 1 means horrible alignment of handwriting, e.g. if it's super sloped. "
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]

        
        )
        
    except Exception as e:
        Messagebox.show_error(f"Error: {e}")
        return

    list_response2 = response2.choices[0].message.content.strip().split(',')

    XInputs = list_response1 + list_response2

    return XInputs






class Application(ttk.Window):
    def __init__(self):
        super().__init__()

        self.title("Dyslexia Detection Application")
        self.geometry("1800x1200")

        # Shared data
        self.text_data = ''

        style = ttk.Style()
        style.theme_use('superhero')

        #initialise widgets. 
        self.create_widgets()

    def create_widgets(self): 
        global frames

        # Create sidebar
        self.sidebar = ttk.Frame(self, bootstyle="primary", width=200)
        self.sidebar.pack(side=LEFT, fill=BOTH)

        # Logo or App Title
        self.logo_label = ttk.Label(self.sidebar, text="DyslexAI", font=("Helvetica", 20, "bold"), bootstyle="inverse-primary")
        self.logo_label.pack(pady=20)

        # Navigation Buttons
        self.nav_buttons = {}
        buttons = [("Home", self.show_home),
                   ("Upload", self.show_upload),
                   ("Analysis", self.show_analysis),
                  ("Help", self.show_help)]

        for (text, command) in buttons:
            btn = ttk.Button(self.sidebar, text=text, command=command, bootstyle="outline-light")
            btn.pack(pady=5, fill=X, padx=10)
            self.nav_buttons[text] = btn
        
        self.container = ttk.Frame(self)
        self.container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # for F in (HomePage, UploadPage, AnalysisPage, HelpPage):
        frames = {}
        for F in (HomePage, UploadPage, AnalysisPage):
            page_name = F.__name__
            frame = F(parent=self.container, controller=self)
            frames[page_name] = frame
            frame.grid(row=0, column=0, sticky=NSEW)
            if F == AnalysisPage: 
                frame.create_widgets()
                

            # the problem here is creting the object of AnalysisClass but before finishing creating that, it is asking frames["AnalysisPage"].result_labesl
            #but there is no AnalysisPage in frames yet! 

        

        self.show_frame("HomePage")

    def show_frame(self, page_name):
        frame = frames[page_name]
        frame.tkraise()

    def show_home(self):
        self.show_frame("HomePage")

    def show_upload(self):
        self.show_frame("UploadPage")

    def show_analysis(self):
        self.show_frame("AnalysisPage")

    def show_help(self):
        self.show_frame("HelpPage")

    def on_closing(self):
        self.destroy()



class HomePage(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(padding=20)
        self.create_widgets()

    def create_widgets(self):
        # Title
        title_label = ttk.Label(self, text="Welcome to DislexAI", font=("Helvetica", 24, "bold"), anchor="center", justify="center")
        title_label.pack(pady=20, fill="x", expand=True)

        # Description
        desc = "Empowering educators and individuals to detect Dyslexia early through Machine Learning."
        title_label = ttk.Label(self, text=desc, font=("Helvetica", 14), anchor="center", justify="center")
        title_label.pack(pady=10, fill="x", expand=True)

        # Placeholder for Image or Animation
        # You can add an image here if desired
        # Example:
        # img = ImageTk.PhotoImage(Image.open("path_to_image.png").resize((300, 300)))
        # img_label = ttk.Label(self, image=img)
        # img_label.image = img
        # img_label.pack(pady=20)

class UploadPage(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.img = None
        self.configure(padding=20)
        self.create_widgets()

    def create_widgets(self):

        # Upload Area
        title_label = ttk.Label(self, text="Upload A Handwriting Sample (~ 3 sentences)", font=("Helvetica", 18, "bold"), anchor="center", justify="center")
        title_label.pack(pady=40, fill="x", expand=True)

        # Select Image Button
        select_btn = ttk.Button(self, text="Select Handwriting Sample Image", command=self.select_image, bootstyle="success")
        select_btn.pack(pady=40)

        # File Name Display
        self.file_name_var = tk.StringVar(value="No file selected")
        file_name_label = ttk.Label(self, textvariable=self.file_name_var, font=("Helvetica", 12))
        file_name_label.pack(pady=5)

        # Image Preview
        self.image_preview = ttk.Label(self, bootstyle="secondary", anchor="center")
        self.image_preview.pack(pady=10, padx=50, fill=BOTH, expand=True)

        # Analysis Messages
        self.analysis_message_var = tk.StringVar(value="")
        analysis_msg_label = ttk.Label(self, textvariable=self.analysis_message_var, font=("Helvetica", 12), wraplength=800, justify="center")
        analysis_msg_label.pack(pady=5)

        self.analysis_message_var2 = tk.StringVar(value="")
        analysis_msg_label2 = ttk.Label(self, textvariable=self.analysis_message_var2, font=("Helvetica", 12), wraplength=800, justify="center")
        analysis_msg_label2.pack(pady=5)

    def select_image(self):
        global image_path
        image_path = filedialog.askopenfilename(title="Select an Image File",
                                               filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;"), ("All Files", "*.*")])
        if image_path:
            self.display_and_label_image(image_path)

    def display_and_label_image(self, image_path):
        global XInputs
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            self.image_path = image_path
            self.image_type = image_path.split('.')[-1]
            file_name = os.path.basename(image_path)
            self.file_name_var.set(f"Selected File: {file_name}")

            # Load and display image
            try:
                img = Image.open(image_path)
                img.thumbnail((400, 400))
                self.img = ImageTk.PhotoImage(img)
                self.image_preview.configure(image=self.img, borderwidth=2, relief="groove")
            except Exception as e:
                Messagebox.show_error("Invalid image", "The selected file is not a valid image.")
                self.image_preview.configure(image='')
                self.file_name_var.set("No file selected")
                self.analysis_message_var.set("")
                self.analysis_message_var2.set("")
                return

            self.analysis_message_var.set("Valid image, starting Machine Learning Inference")
            self.analysis_message_var2.set("")

            XInputs = label_image(image_path)
            


        else:
            Messagebox.show_warning("Invalid File", "Please select a valid image file.")

class AnalysisPage(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.analysis_results = None
        self.current_factor = '% corrections to words'  # Default factor
        self.configure(padding=20)

    def create_widgets(self):
        main_layout = ttk.Frame(self)
        main_layout.pack(fill=BOTH, expand=True, padx=10, pady=10)

        # Title
        title_label = ttk.Label(main_layout, text="Image Analysis", font=("Helvetica", 20, "bold"))
        title_label.pack(pady=10)

        # Start Analysis Button
        self.start_analysis_btn = ttk.Button(main_layout, text="Start Analysis", command=self.start_analysis, bootstyle="success")
        self.start_analysis_btn.pack(pady=10)

        # Progress Bar
        self.progress = ttk.Progressbar(main_layout, mode='determinate', length=400)
        self.progress.pack(pady=10)

        # Analysis Results Sections
        results_frame = ttk.Frame(main_layout)
        results_frame.pack(pady=10, fill=BOTH, expand=True)

        # Left Side: Graph and Factor Buttons
        left_frame = ttk.Frame(results_frame)
        left_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=10)

        # Factor Selection
        factors = ['% corrections to words', '% spelling error', '% case mistakes', 'joining', 'ledgibility', 'line alignment']
        self.factor_var = tk.StringVar(value=self.current_factor)

        button_frame = ttk.LabelFrame(left_frame, text="Select Factor")
        button_frame.pack(pady=5)

        for factor in factors:
            btn = ttk.Radiobutton(button_frame, text=factor, variable=self.factor_var, value=factor, command=self.change_factor, bootstyle="info")
            btn.pack(side=LEFT, padx=2)

        # Graph Title
        self.graph_title = ttk.Label(left_frame, text=self.current_factor, font=("Helvetica", 14, "bold"))
        self.graph_title.pack(pady=5)

        #create a matplotlib figure: 
        self.figure = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)  # Create an Axes object
        self.canvas = FigureCanvasTkAgg(self.figure, master=left_frame)
        self.canvas.get_tk_widget().pack(fill = BOTH, expand=  True)       

        # # Graph Area
        # self.figure = plt.Figure(figsize=(5, 4), dpi=100)
        # self.ax = self.figure.add_subplot(111)
        # self.canvas = ttk.canvas(self, master=left_frame)
        # self.canvas = FigureCanvasTkAgg(self.figure, master=left_frame)
        # self.canvas.draw()
        # self.canvas.get_tk_widget().pack()

        
        # Right Side: Results
        right_frame = ttk.Frame(results_frame)
        right_frame.pack(side=RIGHT, fill=BOTH, expand=True, padx=10)

        # Percentages and Results Frame
        results_inner_frame = ttk.LabelFrame(right_frame, text="Analysis Results")
        results_inner_frame.pack(fill=BOTH, expand=True)

        # Percentages Labels
        for factor in factors:
            frame = ttk.Frame(results_inner_frame)
            frame.pack(anchor="w", padx=10, pady=2)
            lbl = ttk.Label(frame, text=f"{factor}:", font=("Helvetica", 12, "bold"))
            lbl.pack(side=LEFT)
            val = ttk.Label(frame, text="N/A", font=("Helvetica", 12))
            val.pack(side=LEFT, padx=5)
            frames["AnalysisPage"].result_labels = getattr(self, 'result_labels', {})
            self.result_labels[factor] = val

        # Dyslexia Probability
        dys_label = ttk.Label(results_inner_frame, text="% Chance of Dyslexia:", font=("Helvetica", 14, "bold"))
        dys_label.pack(pady=10)
        self.dyslexia_var = tk.StringVar(value="N/A")
        dys_value = ttk.Label(results_inner_frame, textvariable=self.dyslexia_var, font=("Helvetica", 12))
        dys_value.pack()

    def start_analysis(self): # this is run by the start_analysis button. 
        upload_page = frames["UploadPage"]
        if not image_path:
            Messagebox.show_warning("No Image", "Please upload an image in the Upload section.")
            return

        self.start_analysis_btn.state(['disabled'])
        self.progress['value'] = 0
        self.run_analysis()

    def run_analysis(self):
        global XInputs, XInputsNumpy
        # Simulate analysis time
        for i in range(1, 101):
            time.sleep(0.01)
            self.progress['value'] = i
            self.update_idletasks()

        # Perform analysis

        #finding their probability of dyslexia: 
        X_Vals = {'% corrections to words': float(XInputs[0]), 
              '% spelling error':float(XInputs[1]), 
              '% case mistakes':float(XInputs[2]), 
              'joining':float(XInputs[3]),
              'ledgibility':float(XInputs[4]),
              'line alignment':float(XInputs[5])
              
             }
        X_df = pd.DataFrame(X_Vals, index = [0])
        XInputsNumpy = X_df.to_numpy()
        XInputsNumpy = XInputsNumpy.reshape(1, -1)

        dyslexia_prob = float(LogRegModel.predict_proba(XInputsNumpy)[:,1][0]) * 100


        # Prepare results
        your_results = {}
        factors = [r'% corrections to words', r'% spelling error', r'% case mistakes', 'joining', 'ledgibility', 'line alignment']
        for idx, factor in enumerate(factors):
            your_results[factor] = XInputs[idx]

        self.analysis_results = {
            'your_results': your_results
        }

        print("your results:")
        print(self.analysis_results)

        # Update GUI with results
        self.after(0, self.update_results, dyslexia_prob)

    def update_results(self, dyslexia_probability):
        # Update "Your Results" labels
        for factor, value in self.analysis_results['your_results'].items():
            self.result_labels[factor].config(text=f"{round(float(value), 3)}")

        self.dyslexia_var.set(f"{round(float(dyslexia_probability), 3)}%")

        # Display graph for the current factor
        self.show_graph(self.current_factor)

        self.start_analysis_btn.state(['!disabled'])

    def change_factor(self):
        self.current_factor = self.factor_var.get()
        self.graph_title.config(text=self.current_factor)
        self.show_graph(self.current_factor)

    def show_graph(self, factor):
        if not hasattr(self, 'analysis_results'):
            Messagebox.show_warning("No Results", 'Please perform analysis first - Click "Start Analysis".')
            return

        your_result = float(self.analysis_results['your_results'].get(factor, 0.0))

        # Clear the previous plot
        self.ax.clear()

        # Plot training datapoints
        dyslexic = df['dyslexic']
        print(f"This is what the dyslexic variable holds: {dyslexic}")
        y_values = df[factor]
        colors = ['green' if x == 0 else 'red' for x in dyslexic]
        self.ax.scatter(range(1, len(y_values)+1), y_values, c=colors, label='Training datapoints (red=dyslexic, green=non-dyslexic)')

        # Plot "Your Result" line
        self.ax.axhline(y=your_result, color='blue', linestyle='-', label='Your Result')

        self.ax.set_xlabel("Training Datapoint Number")
        self.ax.set_ylabel(f"{factor} (%)")
        self.ax.set_title(factor.capitalize())
        self.ax.legend()

        self.figure.tight_layout()
        self.canvas.draw()







# class SettingsPage(ttk.Frame):
#     def __init__(self, parent, controller):
#         super().__init__(parent)
#         self.controller = controller
#         self.configure(padding=20)
#         self.create_widgets()

#     def create_widgets(self):
#         layout = ttk.Frame(self)
#         layout.pack(pady=20, fill=BOTH, expand=True)

#         # Title
#         title_label = ttk.Label(layout, text="Settings", font=("Helvetica", 20, "bold"))
#         title_label.pack(pady=10)



#         available_themes = ['darkly', 'lightly', 'cosmo', 'superhero', 'yeti', 'morph']
#         self.theme_var = tk.StringVar(value=self.controller.current_theme)
#         self.theme_combo = ttk.Combobox(layout, textvariable=self.theme_var, values=available_themes, state='readonly', bootstyle="info")
#         self.theme_combo.pack(pady=5)
#         self.theme_combo.bind("<<ComboboxSelected>>", self.change_theme)

#         # Save Settings Button
#         save_btn = ttk.Button(layout, text="Save Settings", command=self.save_settings, bootstyle="success")
#         save_btn.pack(pady=20)

#     def adjust_font_size(self):
#         new_size = self.font_size_var.get()
#         self.controller.font_size = new_size
#         style = ttk.Style()
#         style.configure('.', font=("Helvetica", self.controller.font_size))

#     def change_theme(self, event):
#         selected_theme = self.theme_var.get()
#         self.controller.current_theme = selected_theme
#         self.controller.style.theme_use(selected_theme)

#     def save_settings(self):
#         Messagebox.show_info("Settings Saved", "Your settings have been saved successfully.")

# class HelpPage(ttk.Frame):
#     def __init__(self, parent, controller):
#         super().__init__(parent)
#         self.controller = controller
#         self.configure(padding=20)
#         self.create_widgets()

#     def create_widgets(self):
#         layout = ttk.Frame(self)
#         layout.pack(pady=10, padx=10, fill=BOTH, expand=True)

#         # Title
#         title_label = ttk.Label(layout, text="Help and Feedback", font=("Helvetica", 20, "bold"))
#         title_label.pack(pady=10)

#         # FAQs
#         faq_label = ttk.Label(layout, text="Frequently Asked Questions", font=("Helvetica", 16))
#         faq_label.pack(pady=5)

#         faq_content = """
#         Q: How does the dyslexia detection work?
#         A: Our application uses advanced image processing techniques to analyze visual patterns that may indicate dyslexia.

#         Q: Is my data secure?
#         A: Yes, your data is processed securely and not stored permanently.

#         Q: Can I adjust the analysis settings?
#         A: Yes, you can adjust the model parameters in the Settings page.
#         """

#         faq_text = tk.Text(layout, wrap="word", height=10, font=("Helvetica", 12))
#         faq_text.insert("1.0", faq_content)
#         faq_text.configure(state="disabled", bg=self.cget("background"))
#         faq_text.pack(pady=5, fill=BOTH, expand=True)

#         # Feedback Submission
#         feedback_label = ttk.Label(layout, text="Submit Feedback", font=("Helvetica", 16))
#         feedback_label.pack(pady=10)

#         self.feedback_input = ttk.Text(layout, height=5, font=("Helvetica", 12))
#         self.feedback_input.pack(pady=5, fill=BOTH, expand=True)

#         submit_feedback_btn = ttk.Button(layout, text="Submit Feedback", command=self.submit_feedback, bootstyle="success")
#         submit_feedback_btn.pack(pady=5)

#     def submit_feedback(self):
#         feedback = self.feedback_input.get("1.0", tk.END).strip()
#         if feedback:
#             # Handle feedback submission (e.g., save to file or send to server)
#             try:
#                 with open("feedback.txt", "a") as f:
#                     f.write(feedback + "\n\n")
#                 self.feedback_input.delete("1.0", tk.END)
#                 Messagebox.show_info("Thank You", "Thank you for your feedback!")
#             except Exception as e:
#                 Messagebox.show_error("Error", f"Failed to save feedback: {e}")
#         else:
#             Messagebox.show_warning("No Feedback", "Please enter your feedback before submitting.")



app = Application()
app.mainloop()

