# %pip install sentence_transformers
# %pip install language-tool-python 
import sys
# import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from difflib import SequenceMatcher
import diff_text
# import language_tool_python

# A = 0.25
# B = 0.25
# C = 0.25
# D = 0.25 
OPTIMAL_LENGTH = 0.7


def evaluate_on_meaning(original_paragraph, response):
  '''
  1st possible evaluate function that checks the semantic closeness of the response
  to the original sentence; Could be used to infer whether important words are removed
  Returns: a float (cosine similarity value)
  '''
  mpnet = SentenceTransformer('all-mpnet-base-v2') # SOTA model, better than SBERT
  embedding_original = mpnet.encode(original_paragraph)
  embedding_response = mpnet.encode(response)
  return cos_sim(embedding_original, embedding_response).item()


def evaluate_on_length(original_paragraph, response):
  '''
  2nd possible evaluate function that checks the lengths of the shortened sentence
  Could be used to infer whether unnecessary phrases are indeed removed
  Returns: a float (length shortened/length original)
  '''
  return 1 - abs(len(response)/len(original_paragraph) - OPTIMAL_LENGTH)


def evaluate_on_paraphrasing(original_paragraph, response):
  '''
  3rd possible evaluate function that checks the occurences of paraphrasing on a word level
  Returns: a float (# of non-occurences/length original)
  '''
   # Split them into words by whitespace, so we diff on words instead of letters:
  # p1 = original_paragraph.split()
  # p2 = response.split()
  # # Diff on words, using difflib:
  # s = SequenceMatcher(None, p1, p2)
  opcodes = diff_text.diff_text(original_paragraph, response, False)
  rst = 0
  for code in opcodes:
    if code[0] in ['insert', 'replace']:
      rst += 1
  return 1 - rst/len(original_paragraph.split())


# def evaluate_on_grammaticality(response):
#   '''
#   4th possible evaluate function that checks whether the shortened sentence is grammatical
#   Returns: 1 if grammatical, 0 otherwise
#   '''
#   checker = language_tool_python.LanguageTool('en-US')
#   matches = checker.check(response)
#   # checker.close()
#   for match in matches:
#     if match.ruleId not in ['UPPERCASE_SENTENCE_START']:
#       return 0
#   return 1

def composite(original_paragraph, response):
  # print('The composite score is ' + str(A*evaluate_on_meaning(original_paragraph, response) + B*evaluate_on_length(original_paragraph, response) + C*evaluate_on_paraphrasing(original_paragraph, response) + D* evaluate_on_grammaticality(response)))
  return evaluate_on_meaning(original_paragraph, response) + evaluate_on_length(original_paragraph, response) + evaluate_on_paraphrasing(original_paragraph, response)

def revert_paraphrasing(original_paragraph, response):
  p1 = original_paragraph.split()
  p2 = response.split()
  # Diff on words, using difflib:
  # s = SequenceMatcher(None, p1, p2)
  opcodes = diff_text.diff_text(original_paragraph, response, False)
  rst = ''
  for code in opcodes:
    if code[0] == 'equal':
      rst += (' '.join(p2[code[3]:code[4]]) + ' ')
    elif code[0] == 'replace':
      rst += (' '.join(p1[code[1]:code[2]]) + ' ')
  return rst

if __name__ == "__main__":
  shortened = str(sys.argv[1])
  original = str(sys.argv[2])
  print(composite(original, shortened))