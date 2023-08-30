# import speech_recognition as sr
#
# def real_time_speech_recognition():
#     recognizer = sr.Recognizer()
#     microphone = sr.Microphone()
#
#     print("Listening for speech...")
#
#     with microphone as source:
#         recognizer.adjust_for_ambient_noise(source)  # Adapt to ambient noise
#         audio = recognizer.listen(source)  # Listen for audio
#
#     print("Recognizing...")
#
#     try:
#         recognized_text = recognizer.recognize_google(audio)  # Use Google Web Speech API
#         print("Recognized: ", recognized_text)
#     except sr.UnknownValueError:
#         print("Sorry, I could not understand audio.")
#     except sr.RequestError as e:
#         print("Sorry, I couldn't request results from Google Web Speech API; {0}".format(e))
#
# if __name__ == "__main__":
#     real_time_speech_recognition()


import speech_recognition as sr

# Initialize the recognizer
recognizer = sr.Recognizer()

# Open the microphone for capturing audio
with sr.Microphone() as source:
    print("Say something...")

    try:
        while True:
            # Listen to the user's speech
            audio = recognizer.listen(source)

            # Recognize the speech using Google Web Speech API
            text = recognizer.recognize_google(audio)

            print("You said:", text)

    except KeyboardInterrupt:
        print("Stopping the speech recognition.")
    except sr.UnknownValueError:
        print("Could not understand audio.")
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))