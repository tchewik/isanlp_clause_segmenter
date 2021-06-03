import conllu
import numpy as np
from isanlp.annotation_rst import DiscourseUnit
from isanlp.utils.annotation_conll_converter import AnnotationCONLLConverter

from catboost_clf import CatBoostClf
from feature_extractor import FeatureExtractor


class Processor:
    def __init__(self, model_dir_path):
        self._model_dir_path = model_dir_path
        self._conll_converter = AnnotationCONLLConverter()
        self._feature_extractor = FeatureExtractor()
        self._model = CatBoostClf(model_dir_path)
        print('Initialized.')

    def __call__(self, annot_text, annot_tokens, annot_sentences, annot_lemma, annot_morph, annot_postag,
                 annot_syntax_dep_tree):
        print('received call')
        annotation = {
            'text': annot_text,
            'tokens': annot_tokens,
            'sentences': annot_sentences,
            'lemma': annot_lemma,
            'morph': annot_morph,
            'ud_postag': annot_postag,
            'syntax_dep_tree': annot_syntax_dep_tree
        }
        print(annotation)

        converted_annot = ""
        for line in self._conll_converter(doc_id='0', annotation=annotation):
            converted_annot += line + '\n'

        print(converted_annot)

        sentences = conllu.parse(converted_annot)
        print(sentences)
        features = self._feature_extractor(sentences)
        print(features)
        predictions = np.argwhere(np.array(self._model.predict(features)) == 1)[:, 0]
        print(predictions)
        return self._build_discourse_units(annot_text, annot_tokens, predictions)

    def _build_discourse_units(self, text, tokens, numbers):
        """
        :param text: original text
        :param list tokens: isanlp.annotation.Token
        :param numbers: positions of tokens predicted as EDU left boundaries (beginners)
        :return: list of DiscourseUnit
        """

        edus = []
        start_id = 0

        if numbers.shape[0]:
            for i in range(0, len(numbers) - 1):
                new_edu = DiscourseUnit(start_id + i,
                                        start=tokens[numbers[i]].begin,
                                        end=tokens[numbers[i + 1]].begin - 1,
                                        text=text[tokens[numbers[i]].begin:tokens[numbers[i + 1]].begin],
                                        relation='elementary',
                                        nuclearity='_')
                edus.append(new_edu)

            if numbers.shape[0] == 1:
                i = -1

            new_edu = DiscourseUnit(start_id + i + 1,
                                    start=tokens[numbers[-1]].begin,
                                    end=tokens[-1].end,
                                    text=text[tokens[numbers[-1]].begin:tokens[-1].end],
                                    relation='elementary',
                                    nuclearity='_')
            edus.append(new_edu)

        return edus
