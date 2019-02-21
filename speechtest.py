from gtts import gTTS
import pyttsx
state= True
while state:
    print("Input: ")
    x= input()
    print(x)
    if x=="0":
        state=False
    #y=gTTS(text=x, lang='en')
    #y.save('sample.mp3')

    engine = pyttsx.init()
    engine.say(x)
    engine.runAndWait()
    
    
    
