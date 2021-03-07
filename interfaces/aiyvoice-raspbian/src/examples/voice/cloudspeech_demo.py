#!/usr/bin/env python3
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

"""A demo of the Google CloudSpeech recognizer."""
import argparse
import locale
import logging

from aiy.voice import tts
from aiy.board import Board, Led
from aiy.cloudspeech import CloudSpeechClient


def get_hints(language_code):
    if language_code.startswith('en_'):
        return ('turn on the light',
                'turn off the light',
                'blink the light',
                'goodbye')
    return None

def locale_language():
    language, _ = locale.getdefaultlocale()
    return language

def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(description='Assistant service example.')
    parser.add_argument('--language', default=locale_language())
    args = parser.parse_args()

    logging.info('Initializing for language %s...', args.language)
    hints = get_hints(args.language)
    client = CloudSpeechClient()
    with Board() as board:
        while True:
            if hints:
                logging.info('Say something, e.g. %s.' % ', '.join(hints))
            else:
                logging.info('Say something.')
            text = client.recognize(language_code=args.language,
                                    hint_phrases=hints)
            if text is None:
                logging.info('You said nothing.')
                continue

            logging.info('You said: "%s"' % text)
            text = text.lower()
            if 'turn on the light' in text:
                board.led.state = Led.ON
            elif 'turn off the light' in text:
                board.led.state = Led.OFF
            elif 'blink the light' in text:
                board.led.state = Led.BLINK
            elif 'repeat after me' in text:
                # Remove "repeat after me" from the text to be repeated
                to_repeat = text.replace('repeat after me', '', 1)
                tts.google_tts_say(to_repeat)
	    # some Godfather quotes
            elif 'ask for justice' in text:
                tts.google_tts_say('The court gave you justice.', gender='MALE')
            elif 'an eye for an eye' in text:
                tts.google_tts_say('But your daughter is still alive.', gender='MALE')
            elif 'how much shall i pay you' in text or 'how much should i pay you' in text:
                tts.google_tts_say("""You never think to protect yourself with real friends. 
                You think it's enough to be an American. All right, the Police protects you, there are Courts of Law, so you don't need a friend like me.
		        But now you come to me and say Don Corleone, you must give me justice. And you don't ask in respect or friendship.
                And you don't think to call me Godfather; instead you come to my house on the day my daughter is to be married and you ask me to do murder for money.""", gender='MALE', type='ssml')
            elif 'goodbye' in text:
                break

if __name__ == '__main__':
    main()
