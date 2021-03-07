# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
An API that performs text-to-speech.
You can also use this to perform text-to-speech from the command line::
    python ~/AIY-projects-python/src/aiy/voice/tts.py "hello world"
"""

import argparse
import os
import subprocess
import tempfile
from google.cloud import texttospeech
import pygame 
import html

RUN_DIR = '/run/user/%d' % os.getuid()
# Instantiates a client
client = texttospeech.TextToSpeechClient()

def say(text, lang='en-US', volume=60, pitch=130, speed=100, device='default'):
    """
    Speaks the provided text.
    Args:
        text: The text you want to speak.
        lang: The language to use. Supported languages are:
            en-US, en-GB, de-DE, es-ES, fr-FR, it-IT.
        volume: Volume level for the converted audio. The normal volume level is
            100. Valid volume levels are between 0 (no audible output) and 500 (increasing the
            volume by a factor of 5). Values higher than 100 might result in degraded signal
            quality due to saturation effects (clipping) and is not recommended. To instead adjust
            the volume output of your device, enter ``alsamixer`` at the command line.
        pitch: The pitch level for the voice. The normal pitch level is 100, the allowed values lie
            between 50 (one octave lower) and 200 (one octave higher).
        speed: The speed of the voice. The normal speed level is 100, the allowed values lie
            between 20 (slowing down by a factor of 5) and 500 (speeding up by a factor of 5).
        device: The PCM device name. Leave as ``default`` to use the default ALSA soundcard.
    """
    data = "<volume level='%d'><pitch level='%d'><speed level='%d'>%s</speed></pitch></volume>" % \
           (volume, pitch, speed, text)
    with tempfile.NamedTemporaryFile(suffix='.wav', dir=RUN_DIR) as f:
       cmd = 'pico2wave --wave %s --lang %s "%s" && aplay -q -D %s %s' % \
             (f.name, lang, data, device, f.name)
       subprocess.check_call(cmd, shell=True)

def text_to_ssml(inputfile):

    raw_lines = inputfile

    # Replace special characters with HTML Ampersand Character Codes
    # These Codes prevent the API from confusing text with
    # SSML commands
    # For example, '<' --> '&lt;' and '&' --> '&amp;'

    escaped_lines = html.escape(raw_lines)

    # Convert plaintext to SSML
    # Wait two seconds between each address
    ssml = "<speak>{}</speak>".format(
        escaped_lines.replace("\n", '\n<break time="2s"/>')
    )

    # Return the concatenated string of ssml script
    return ssml

def google_tts_say(text, lang='en-US',gender='NEUTRAL',type='text'):
    """
    Speaks the provided text.
    Args:
        text: The text you want to speak.
        lang: The language to use. Supported languages are:
            en-US, en-GB, de-DE, es-ES, fr-FR, it-IT.
        gender: gender
    """
    if type == 'ssml':
        text = text_to_ssml(text)

    if gender == 'NEUTRAL':
        g = texttospeech.SsmlVoiceGender.NEUTRAL
    elif gender == 'MALE':
        g = texttospeech.SsmlVoiceGender.MALE
    elif gender == 'FEMALE':
        g = texttospeech.SsmlVoiceGender.FEMALE
    

    # Set the text input to be synthesized
    if type == 'ssml':
        synthesis_input = texttospeech.SynthesisInput(ssml=text)
    else:
        synthesis_input = texttospeech.SynthesisInput(text=text)

    # Build the voice request, select the language code ("en-US") and the ssml
    # voice gender ("neutral")
    voice = texttospeech.VoiceSelectionParams(
        language_code=lang, ssml_gender=g
    )

    # Select the type of audio file you want returned
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    # Perform the text-to-speech request on the text input with the selected
    # voice parameters and audio file type
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # The response's audio_content is binary.
    with open("output.mp3", "wb") as out:
        # Write the response to the output file.
        out.write(response.audio_content)
        print('Audio content written to file "output.mp3"')

    pygame.init()
    pygame.mixer.music.load('output.mp3')
    pygame.mixer.music.play()
    print('playing')
    while pygame.mixer.music.get_busy() == True:
        continue


def _main():
    parser = argparse.ArgumentParser(description='Text To Speech (pico2wave)')
    parser.add_argument('--lang', default='en-US')
    parser.add_argument('--volume', type=int, default=60)
    parser.add_argument('--pitch', type=int, default=130)
    parser.add_argument('--speed', type=int, default=100)
    parser.add_argument('--device', default='default')
    parser.add_argument('text', help='path to disk image file ')
    args = parser.parse_args()
    google_tts_say(args.text)
    #say(args.text, lang=args.lang, volume=args.volume, pitch=args.pitch, speed=args.speed,
    #    device=args.device)


if __name__ == '__main__':
    _main()
