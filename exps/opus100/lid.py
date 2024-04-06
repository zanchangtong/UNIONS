
import fasttext
import argparse

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", type=str, default="None", \
        help="The path of source file")
    parser.add_argument("--lang", type=str, default='en', \
        help="The language to identification")
    return parser.parse_args()


class LanguageIdentification:

    def __init__(self):
        pretrained_lang_model = "../lid.176.bin"
        self.model = fasttext.load_model(pretrained_lang_model)

    def predict_lang(self, text):
        predictions = self.model.predict(text, k=1) # returns top 1 matching languages
        return predictions

if __name__ == '__main__':
    args = parse_config()
    src_path = args.src_path
    lang = args.lang
    LANGUAGE = LanguageIdentification()

    with open(src_path, 'r', encoding='utf-8') as f2:
        total_num=0
        i = 0
        for line in f2:
            total_num += 1
            language = LANGUAGE.predict_lang(line.strip())
            l = language[0][0].split('_')[-1]
            if l == lang:
                i+=1

    percentage=i/total_num
    print(1-percentage)



