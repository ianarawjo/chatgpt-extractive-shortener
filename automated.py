"""
    This Python script takes a paragraph as input, 
    and runs it successively through ChatGPT to extract a shortened version
    that performs "nearly-extractive" summarization: a version with deleted words/phrases that don't
    contribute to the overall meaning, and minimal word changes or additions.

    The algorithm is as follows:
        0. We choose N (number of responses per query to ChatGPT), 
           and D, the "max depth", or how many times we'll repeat the loop and successively shorten.
           d, the current depth, is set to 0.
        1. We collect N responses from ChatGPT for each paragraph P input, for some variety. 
        2. We run each response R through a diff-checker to detect how many adding/changed words
          there were (since ChatGPT is imperfect in rigidly sticking to our instructions).
        3. We pick the response R_min that:
            - minimizes the number of added/changed words, and 
            - maximizes the number of deletions.
        4. Store the response at depth 'd'. Increment 'd' by one.
        5. If 'd' < 'D', run R_min back through ChatGPT as input (set P=R_min and goto Step 1).
"""
import openai, sys, os, re, json
import argparse
from collections import Counter
from termcolor import colored, RESET
from difflib import SequenceMatcher
from promptengine.pipelines import PromptPipeline
from promptengine.template import PromptTemplate, PromptPermutationGenerator
from promptengine.utils import LLM, extract_responses, is_valid_filepath
from diff_text import diff_text
import eval_response

EXTRACTIVE_SHORTENER_PROMPT_TEMPLATE = \
"""Delete 10 words or phrases from the following paragraph that don't contribute much to its meaning, but keep readability:
"${paragraph}"

Please do not add any new words or change words, only delete words."""

HTML_GRAY_LEVELS = ['#000000', '#767676', '#A0A0A0', '#B5B5B5', '#D0D0D0']
LATEX_GRAY_CODES = ['BLACK', 'GRAY0', 'GRAY1', 'GRAY2', 'GRAY3']

# PromptPipeline that runs the 'extractive shortner' prompt, and cache's responses.
class ExtractiveShortenerPromptPipeline(PromptPipeline):
    def __init__(self):
        self._template = PromptTemplate(EXTRACTIVE_SHORTENER_PROMPT_TEMPLATE)
        storageFile = 'shortened_responses.json'
        super().__init__(storageFile)
    def gen_prompts(self, properties):
        gen_prompts = PromptPermutationGenerator(self._template)
        return list(gen_prompts({
            "paragraph": properties["paragraph"]
        }))

def extract_sentences_from_para(paragraph: str) -> list:
    return re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', paragraph)

def extract_contiguous_sequences(arr):
    sequences = []
    prev_num = None
    cur_sequence = {}
    for i, num in enumerate(arr):
        if i == 0:
            cur_sequence['start'] = i
            cur_sequence['val'] = num
        elif num != prev_num:
            cur_sequence['end'] = i
            sequences.append(cur_sequence)
            cur_sequence = {'start': i, 'val': num}
        prev_num = num
    
    cur_sequence['end'] = len(arr)
    sequences.append(cur_sequence)

    return sequences

# Script start
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='This script performs word-level extractive summarization through ChatGPT, at successive degrees of word importance.')
    parser.add_argument('paragraph', help='Input paragraph/text', type=str)
    parser.add_argument('--depth', help='The number of successive shortenings to perform (rounds of feeding the text through ChatGPT)', type=int, default=3, nargs='?')
    parser.add_argument('--n', help='The number of responses to request from ChatGPT, for *each* query', type=int, default=8, nargs='?')
    parser.add_argument('--temp', help='The temperature for ChatGPT calls', type=float, default=1.0, nargs='?')
    parser.add_argument('--interactive', help="Instead of picking the 'best' response at each depth automatically, you enter the # of the response to choose.", dest='interactive', action='store_true')
    parser.add_argument('--json', help="Whether to output the successive shorteners in a JSON file. Outputs at the level of individual sentences. NOTE: Output is less useful if number of sentences changed between edits.", type=str, default=None, dest='json_output', nargs='?')
    parser.add_argument('--html', help="Outputs a 'graying visualization' of HTML code, with <span>s to color segments of text. (NOTE: Only works for up to five levels of depth.)", dest='html_output', action='store_true')
    parser.add_argument('--no-cache', help="Don't cache the responses or load responses from the cache.", dest='no_cache', action='store_true')
    parser.add_argument('--latex', help="Outputs a 'graying visualization' in LaTeX code with predefined \\textcolor colors GRAY0, GRAY1 etc.", dest='latex_output', action='store_true')
    args = parser.parse_args()

    # The number of responses to request from ChatGPT, for *each* query
    N = args.n

    # The 'max depth', or number of successive times we'll try to shorten
    MAX_DEPTH = args.depth

    # The temperature for ChatGPT calls
    TEMPERATURE = args.temp

    # Whether interactive mode is on
    INTERACTIVE_STEERING = args.interactive is True


    # Double-check JSON filepath is correct, if specific:
    if args.json_output is not None:
        if not is_valid_filepath(args.json_output):
            raise Exception(f"Invalid filepath for JSON output: {args.json_output}")

    if 'paragraph' not in args:
        print("Please provide some text to shorten as the first argument.")
        is_correct = False
    else:
        paragraph = args.paragraph
        
        print(f"\nOkay, I will run the extractive shortener on the paragraph:\n{paragraph}\n")
        print(f"There will be {MAX_DEPTH} rounds of shortening, and {N} queries per round.")
        inp = input("Do you wish to proceed? (Y): ")
        is_correct = inp.strip() == 'Y'

    # Show the user the input paragraph, to double-check it's what they want:
    if not is_correct:
        print("Aborting.")
        exit(1)
    
    orig_paragraph = paragraph[:]

    try:
        openai.api_key = os.environ['OPENAI_API_KEY']
    except KeyError as e:
        openai.api_key = input("Please enter your OpenAI API key: ")

    def strip_wrapping_quotes(s: str) -> str:
        if s[0] == '"': s = s[1:]
        if s[-1] == '"': s = s[0:-1]
        return s
    
    cur_depth = 0

    # The extractive shortener prompt pipeline
    extractive_shortener = ExtractiveShortenerPromptPipeline()

    extractive_shortener.clear_cached_responses()

    # Store the best responses at each depth
    best_responses = []
    best_response_ids = []

    while cur_depth < MAX_DEPTH:
        # print('\n\n')
        # print("*"*50 + f" Depth {cur_depth} of {MAX_DEPTH-1} " + "*"*50)
        # print(f"Running extractive shortener with N={N} on paragraph:\n{paragraph}\n\n")

        # Run the pipeline by calling ChatGPT
        responses = []
        for res in extractive_shortener.gen_responses({"paragraph": paragraph}, LLM.ChatGPT, n=N, temperature=TEMPERATURE):
            responses.extend(extract_responses(res, llm=LLM.ChatGPT))
        
        # Cleanup responses to remove any wrapping double-quotes:
        responses = [strip_wrapping_quotes(r) for r in responses]
        
        # Run the responses through the diff-checker:
        response_infos = []
        for n, response in enumerate(responses):
            opcodes = diff_text(paragraph, response, print_result=False)

            # Calculate the counts of each operation in 'opcodes':
            counts = Counter([op[0] for op in opcodes])  # the first opcode is the tag name, one of: 'equal', 'delete', 'replace', 'insert'

            # Store the info
            response_infos.append({
                "response": response,
                "opcodes": opcodes,
                "counts": counts,
            })
        


        response_infos.sort(key=lambda x: eval_response.composite(paragraph, x["response"]), reverse=True)
        best_response = response_infos[0]
        best_response["response"] = eval_response.revert_paraphrasing(paragraph, best_response["response"])


        # Increment depth:
        cur_depth += 1

        # Now, if cur_depth is less than MAX_DEPTH, the next iteration of the loop will feed the response back into ChatGPT:
        best_responses.append(best_response)
        paragraph = best_response["response"]

    print('\n')
    print('Here are the shortened results:\n')
    for i, item in enumerate(best_responses):
        
        print('Level ' + str(i) + ': ' + item["response"])
        print('\n\n')