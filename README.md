# DyslexAI

##Logistic regression to automate dyslexia detection from a handwriting sample. 

To run version without UI through jupyter notebooks on the IBM Z Mainframe I configured: http://148.100.109.82:38888/lab/tree/DyslexAIJupyter.ipynb 
(jupyter doesn't support GUIs)

The full software with full tkinter ui is available here. 


Features: % word corrections, % spelling mistakes, % case mistakes, letter joining, writing ledgibility and writing alignment. Logistic regression outputs a % chance of dyslexia. 

This is all supercharged by IBM's snapML library, providing fast training and inference on Generalised Linear Models (e.g. LogReg).

More than 90% accuracy on testing data. 




##Data: 

200 images were downloaded (credits: srummanf github account) for training & testing (80/20). It contains dyslexic and non dyslexic handwriting samples. This dataset will grow in the future to enable more accurate predictions.

The raw images were labelled using an LLM, providing values for each of the relevant features taken into account, this enabled a logistic regression model to be trained. 






##Motivation: 


Representing more than 80% of global learning disabilities, dyslexia is by far the most widespread. Around one in ten people suffer from dyslexia. It therefore is a major issue in the education sector, as by extension 1 in 10 students might be affected by it, depriving young students of an enriching academic experience. Even more concerning is that many cases go undiagnosed for too long, delaying critical intervention, especially at younger ages when essential foundations of knowledge are taught.

The goal is clear: to solve the dyslexia diagnosis dilemma. We aim to create a ubiquitous, easy-to-use early intervention tool that helps identify dyslexic tendencies at an early age, ensuring that students receive the support they need sooner rather than later. By focusing on early detection, we can open doors for students who would otherwise struggle without knowing why. Dyslexia diagnosis should be available to all.



##Misc

Note: ML isn't perfect and inaccuracies are present.

Technical Team: Michael Chotoveli (Leader), Louis Liu, Benoit Ben Moubamba, George Gui




___________________________________________________________________
## To run DyslexAIFullUI.py please refer to this guide on commands to execute on your terminal. 

#### First, check python version 
- python3 --version / python --version # (try both methods) if one of them returns a first number of '3', skip, else run the following: 
- sudo apt-get install python3 / brew install python / alternatively download from website https://www.python.org/downloads/

#### Second, make sure pip is installed
- pip3 --version / pip --version # (try both methods) if one of them returns a first number of '3', skip, else run the following: 
- sudo apt-get install python3-pip
- python3 -m pip install --upgrade pip

#### Required Modules
- `openai`  (pip3 install openai)
- `tkinter` (sudo apt-get install python3-tk)
- `ttkbootstrap` (python3 -m pip install ttkbootstrap)
- `snapml`  (pip3 install snapml)
- `matplotlib` (pip3 install matplotlib)
- `pandas` (pip3 install pandas)
- `Pillow` (pip3 install Pillow)


_________________________________________
_________________________________________


_________________________________________
_________________________________________




