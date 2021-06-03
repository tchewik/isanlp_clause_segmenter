from processor import Processor
from isanlp import PipelineCommon


def create_pipeline(delay_init):
    pipeline_default = PipelineCommon([(Processor(model_dir_path='/models'),
                                        ['text', 'tokens', 'sentences', 'lemma', 'morph', 'postag', 'syntax_dep_tree'],
                                        {0: 'clauses'})
                                       ],
                                      name='default')

    return pipeline_default
