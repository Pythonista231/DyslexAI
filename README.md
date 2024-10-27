## DyslexAI
Logistic Regression to automate dyslexia detection from a sample of handwriting. 

##Dyslexia: 
Representing more than 80% of global learning disabilities,
dyslexia is by far the most widespread. Around one in ten
people suffer from dyslexia. It therefore is a major issue in
the education sector, as by extension 1 in 10 students might
be affected by it, depriving young students of an enriching
academic experience. Even more concerning is that many cases 
go undiagnosed for too long, delaying critical intervention,
especially at younger ages when essential foundations of
knowledge are taught.

##Goal:
The goal is clear: to solve the dyslexia diagnosis dilemma. 
We aim to create a ubiquitous, easy-to-use early intervention 
tool that helps identify dyslexic tendencies at an early age, 
ensuring that students receive the support they need sooner rather 
than later. By focusing on early detection, we can open doors for 
students who would otherwise struggle without knowing why. 
Dyslexia diagnosis should be available to all. 

##The solution:
The solution combines several data points, such as word corrections,
spelling mistakes, case mistakes, letter joining, writing ledgibility
and alignment – all coming together in a proprietary logistic regression
model which outputs a percentage chance of dislexya from one's 
handwriting sample. Furthermore, we also utilise state of the art 
multimodel vision LLMs in order to enable us to label the 
handwriting image a user inputs, before we pass it on to our custom ML model.  

This is simple to use and is open to all. Requires a simple handwriting
sample form the user, while enabling input in multiple languages. 
Furthermore it is executed on IBM Z cloud, meaning no compute requirements
are needed. Furthermore, our software provides descriptive 
analytics to further empower the user through graphs/visualisations. 

##Data: 
100 images have been collected from a public dataset (credits: srummanf (github account))
It contains dyslexic and non dyslexic handwriting samples (supervised).
This dataset will grow in the future to enable more accurate predictions. 

#Important:
Note: ML isn't perfect and inaccuracies are present. 


