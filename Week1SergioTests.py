import unittest

from Week1Sergio import split_contents
from unitesting_utils import load_impact_theory_data


class TestSplitContents(unittest.TestCase):
    '''
    Unit test to ensure proper functionality of split_contents function
    '''
    
    def test_split_contents(self):
        import tiktoken
        from llama_index.text_splitter import SentenceSplitter
        
        data = load_impact_theory_data()
                
        subset = data[:3]
        chunk_size = 256
        chunk_overlap = 0
        encoding = tiktoken.encoding_for_model('gpt-3.5-turbo-0613')
        gpt35_txt_splitter = SentenceSplitter(chunk_size=chunk_size, tokenizer=encoding.encode, chunk_overlap=chunk_overlap)
        results = split_contents(subset, gpt35_txt_splitter)
        self.assertEqual(len(results), 3)
        self.assertEqual(len(results[0]), 83)
        self.assertEqual(len(results[1]), 178)
        self.assertEqual(len(results[2]), 144)
        self.assertTrue(isinstance(results, list))
        self.assertTrue(isinstance(results[0], list))
        self.assertTrue(isinstance(results[0][0], str))
unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromTestCase(TestSplitContents))