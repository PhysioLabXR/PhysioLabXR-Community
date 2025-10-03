from neuspell import CnnlstmChecker
class SpellCorrector:
    def __init__(self):
        self.checker = CnnlstmChecker()
        self.checker.from_pretrained()
    def correct_string(self, stringArr, topK=4):
        return self.checker.correct_string(stringArr)[:topK]


if __name__ == '__main__':
    spell_corrector = SpellCorrector()
    print(spell_corrector.correct_string("doine"))