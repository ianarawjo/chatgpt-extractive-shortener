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

GRAMMAR_CHECKER_PROMPT_TEMPLATE = \
"""Score the following paragraph by how grammatical it is.
"${paragraph}"

Answer A for grammatically correct, B for moderately grammatical, and C for bad grammar. Only respond with one letter."""

GRAMMER_SCORE_RULE = {'A': 1, 'B': 0.5, 'C': 0}
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
    
# PromptPipeline that runs the 'grammar checker' prompt, and cache's responses.
class GrammarCheckerPromptPipeline(PromptPipeline):
    def __init__(self):
        self._template = PromptTemplate(GRAMMAR_CHECKER_PROMPT_TEMPLATE)
        storageFile = 'grammar_checks.json'
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

def find_score(score):
    if 'Answer' in score:
        return score[7:] # Skip the Answer part
    return score

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


    # The grammar_checker prompt pipeline
    grammar_checker = GrammarCheckerPromptPipeline()

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

            grammar_scores = []
            # Get grammar score
            grammar_checker.clear_cached_responses()
            for score in grammar_checker.gen_responses({"paragraph": response}, LLM.ChatGPT, n=1):
                grammar_scores.extend(extract_responses(score, llm=LLM.ChatGPT))

            grammar_score = GRAMMER_SCORE_RULE[find_score(grammar_scores[0])]
            # Store the info
            response_infos.append({
                "response": response,
                "reverted": eval_response.revert_paraphrasing(paragraph, response),
                "opcodes": opcodes,
                "counts": counts,
                "grammar_score": grammar_score,
                "composite_score": eval_response.composite(paragraph, response, grammar_score)
            })

        response_infos.sort(key=lambda x: x["composite_score"], reverse=True)
        best_response = response_infos[0]
        best_response["response"] = best_response["reverted"]


        # Increment depth:
        cur_depth += 1

        # Now, if cur_depth is less than MAX_DEPTH, the next iteration of the loop will feed the response back into ChatGPT:
        best_responses.append(best_response)
        paragraph = best_response["response"]

        print('\n')
        print('Round ' + str(cur_depth) + ' Intermediate Results')
        for i, item in enumerate(response_infos):
            print('\n')
            print(item["reverted"])
            print('Grammar Score: ' + str(item["grammar_score"]))
            print('Composite Score: ' + str(item["composite_score"]))
            if i == 0:
                print('(Picked)')


    print('\n')
    print('Here are the shortened results:\n')
    for i, item in enumerate(best_responses):
        
        print('Level ' + str(i) + ': ' + item["response"])
        print('\n\n')

    split_orig_para = orig_paragraph.split()
    normed_orig_para = ' '.join(split_orig_para)  # Now we are sure every 'word' is separated by exactly one space.

    # Get responses in form [(a, b), (b, c), (c, d), so on], so it's easier to compare each successive level.
    response_pairs = [({"response":normed_orig_para}, best_responses[0])]
    if len(best_responses) > 1:
        response_pairs.extend(zip(best_responses, best_responses[1:]))

    # Each *character* of the input text will be assigned a degree of 'salience' corresponding to its depth:
    char_depths = [0] * len(normed_orig_para)  # everything starts at depth 0 (most salient)

    def char_idx_for_word_idx(_word_idx: int, words: list):
        return sum(len(words[i])+1 for i in range(0, _word_idx))

    def del_char_map_range(char_map, _start: int, _end: int):
        # What this function does is better explained with an example.
        # Suppose we have the sentence:
        #   Is it cool
        #   0123456789    <-- current char_map
        # Now we want to remove 'it' to map it to:
        #   Is cool
        #   0123456    <-- char indices of the revised text
        # The char indices must update to point to their place in the revised text:
        #   Is it cool
        #   012xxx3456    <-- char_map with 'x' for 'None' (removed char)
        # Note how the remaining char indices were shifted 'left'.
        del_len = _end - _start + 1
        for i in range(_start, _end+1):
            if i >= len(char_map):
                print("Warning: Shifting char map index left: Index out of range. Skipping...")
                continue
            char_map[i] = None  # None means there's no mapping, bc char has been deleted
        for i in range(_end+1, len(char_map)):
            # We need to update all remaining indices to shift 'left' 
            # to account for the deletions:
            if i >= len(char_map):
                print("Warning: Shifting char map index left: Index out of range. Skipping...")
                continue
            elif char_map[i] is None:
                print("Warning: Shifting char map index left with None value. Skipping...")
                continue
            char_map[i] -= del_len
    
    def insert_char_map_range(char_map: list, _start: int, _len: int):
        # Same as above, except for insertions. 
        # Suppose we have the sentence:
        #   Is cool
        #   0123456    <-- current char_map
        # Now we want to add 'it' in the revised text:
        #   Is it cool
        #   0123456789    <-- char indices of the revised text
        # The char indices must shift 'right' to point to their place in the revised text:
        #   Is cool
        #   0126789    <-- char_map with shifted indices that skip the inserted portion
        # So everything after insertion _start index must be shifted by +3, which is the length of 'it ', the inserted text
        if _len <= 0:
            return  # nothing to insert
        for i in range(_start, len(char_map)):
            if char_map[i] is None:
                print("Warning: Shifting char map index right with None value. Skipping...")
                continue
            char_map[i] += _len
    
    def invert_char_map(char_map: list) -> dict:
        # Inverts the partial relation 'char_map', returning a dict {idx: idx} (int -> int)
        inv_map = {}
        for i,j in enumerate(char_map):
            if j is None:  # drop the deleted chars
                continue
            inv_map[j] = i
        return inv_map
    
    def get_orig_text_char_idx(char_idx: int, *inv_char_maps) -> int:
        # Given a list of (inverted) char_maps (dicts) and a character index for the current text,
        # returns the index of the character in the original text (e.g., at highest depth), if possible.
        # If not possible (e.g., char was added later and so doesn't exist in original text), returns -1.
        if len(inv_char_maps) == 0: return char_idx
        # Follow the inverse char maps back:
        cur_idx = char_idx
        for icm in inv_char_maps:
            if cur_idx not in icm:
                return -1
            cur_idx = icm[cur_idx]
        return cur_idx
    
    def set_depth(_start_idx: int, _end_idx: int, _depth: int):
        for d_idx in range(_start_idx, _end_idx+1):
            char_depths[d_idx] = _depth

    # Loop through shortening pairs (a, b)
    depth = len(response_pairs)
    inv_char_maps = []
    for prev, cur in response_pairs:
        print("-"*20 + f"Depth {MAX_DEPTH-depth}" + "-"*20)
        opcodes = cur["opcodes"]

        split_prev_para = prev["response"].split()
        normed_prev_para = ' '.join(split_prev_para)

        split_cur_para = cur["response"].split()
        normed_cur_para = ' '.join(split_cur_para)

        # A mapping of char indices: from char index of the 'prev' depth's text, 
        # to char index of the 'cur' depth's text. We will invert this partial relation
        # to understand how a char in the lowest depth (shortest response) maps back up to higher levels. 
        char_map_to_cur = [i for i, _ in enumerate(normed_prev_para)]

        for opcode in opcodes:
            # i1, i2 is the beginning and end of 'prev'; j1, j2 is the same for 'cur'
            # Note that indices are of *words* in the tokenized texts: *not* of character locations.
            # To get character locations, we must map back to the original text, which is the text with .split() run on it. 
            (tag, i1, i2, j1, j2) = opcode

            if tag == 'equal':
                # Even if it thinks the words are equal, there could be punctuation marks 
                # within them that have changed/been inserted (the diff'er ignores punctuation)
                # We need to know about this, for the char_map to remain correct.
                for rel_wrd_idx, prev_wrd in enumerate(split_prev_para[i1:i2]):
                    start_char_idx = char_idx_for_word_idx(i1+rel_wrd_idx, split_prev_para)
                    cur_wrd = split_cur_para[j1+rel_wrd_idx]
                    if len(prev_wrd) != len(cur_wrd):  # mismatching 'matched' words
                        wrd_differ = SequenceMatcher(None, prev_wrd, cur_wrd)
                        print(f'found different chars in matched words:  "{prev_wrd}" <--> "{cur_wrd}"')
                        print('   ops:', wrd_differ.get_opcodes())

                        for wrd_opcode in wrd_differ.get_opcodes():
                            if wrd_opcode[0] in ('replace', 'delete'):
                                print('   -->', start_char_idx+wrd_opcode[1], start_char_idx+wrd_opcode[2], normed_orig_para[start_char_idx+wrd_opcode[1]:start_char_idx+wrd_opcode[2]])

                                del_char_map_range(
                                    char_map_to_cur, 
                                    start_char_idx+wrd_opcode[1], 
                                    start_char_idx+wrd_opcode[2]-1
                                )
                            if wrd_opcode[0] in ('replace', 'insert'):  # we also need to account for any inserted text:
                                insert_char_map_range(
                                    char_map_to_cur, 
                                    start_char_idx+wrd_opcode[1], # the starting char index of the original text where the insertion takes place
                                    wrd_opcode[4]-wrd_opcode[3]  # the length of the inserted text, in # of chars
                                )
            elif tag == 'delete':
                # Gray the entire span of chars for the deleted words:
                start_char_idx = char_idx_for_word_idx(i1, split_prev_para)
                end_char_idx = char_idx_for_word_idx(i2, split_prev_para)
                print(f"marked deleted span: {normed_prev_para[start_char_idx:end_char_idx]}", start_char_idx, end_char_idx)
                del_char_map_range(
                    char_map_to_cur, 
                    start_char_idx, 
                    end_char_idx-1
                )
            elif tag == 'replace':
                # Get the character indices for the replaced span of words
                start_char_idx = char_idx_for_word_idx(i1, split_prev_para)
                end_char_idx = char_idx_for_word_idx(i2, split_prev_para)

                resp_start_char_idx = char_idx_for_word_idx(j1, split_cur_para)
                resp_end_char_idx = char_idx_for_word_idx(j2, split_cur_para)

                # Get the replaced range of chars as a string:
                p1 = normed_prev_para[start_char_idx:end_char_idx]
                p2 = normed_cur_para[resp_start_char_idx:resp_end_char_idx]

                print(f"marked replacement: ({p1}) -> ({p2})")

                # We diff on characters, and then gray whatever chars are *NOT* 'equal' according to the differ: 
                char_differ = SequenceMatcher(None, p1, p2)
                for char_opcode in char_differ.get_opcodes():
                    print(char_opcode)
                    if char_opcode[0] in ('replace', 'delete'):
                        print(' -->', start_char_idx+char_opcode[1], start_char_idx+char_opcode[2], normed_orig_para[start_char_idx+char_opcode[1]:start_char_idx+char_opcode[2]])

                        del_char_map_range(
                            char_map_to_cur, 
                            start_char_idx+char_opcode[1], 
                            start_char_idx+char_opcode[2]-1
                        )
                        
                    if char_opcode[0] in ('replace', 'insert'):  # we also need to account for any inserted text:
                        insert_char_map_range(
                            char_map_to_cur, 
                            start_char_idx+char_opcode[1], # the starting char index of the original text where the insertion takes place
                            char_opcode[4]-char_opcode[3]  # the length of the inserted text, in # of chars
                        )
            elif tag == 'insert':
                start_char_idx = char_idx_for_word_idx(i1, split_prev_para)
                resp_start_char_idx = char_idx_for_word_idx(j1, split_cur_para)
                resp_end_char_idx = char_idx_for_word_idx(j2, split_cur_para)-1
                print(f"marked insertion: {normed_cur_para[resp_start_char_idx:resp_end_char_idx+1]}")
                insert_char_map_range(
                    char_map_to_cur, 
                    start_char_idx, # the starting char index of the original text where the insertion takes place
                    resp_end_char_idx-resp_start_char_idx+1  # the length of the inserted text, in # of chars
                )
        
        # We now have a partial mapping 'char_map_to_cur' from the indices of chars in the 'prev' text, to the 'cur' (revised) text.
        # We can use this 1-level deep mapping to apply the depth mask to the original text, using previous inv_char_maps
        for i, j in enumerate(char_map_to_cur):
            if j is None:  # this char was deleted, so mark it
                orig_char_idx = get_orig_text_char_idx(i, *reversed(inv_char_maps))
                if orig_char_idx > -1:
                    set_depth(orig_char_idx, orig_char_idx, depth)
                else:
                    print(f" --- could not follow char {normed_prev_para[i]} at idx, depth: {i}, {MAX_DEPTH - depth}")

        # We also need to invert this mapping for it to be useful later on:
        inv_char_maps.append(
            invert_char_map(char_map_to_cur)
        )
        
        depth -= 1

    # OPTIONAL: JSON output
    if args.json_output:
        outpath = args.json_output

        # Extract sentences for each response, at each depth:
        resp_sentences = []
        all_responses = [{'response': orig_paragraph}] + best_responses
        for n, r in enumerate(all_responses):
            resp_sentences.append(extract_sentences_from_para(r['response']))
        
        # Check if a sentence was deleted between shortening rounds.
        num_sentences = len(resp_sentences[0])
        if any(len(sentences) != num_sentences for sentences in resp_sentences[1:]):
            print("Warning: The number of sentences changed between shortening rounds. JSON output will be inaccurate.")

        # Need to convert to form: 
        # [{"0":"...", "1":..., "2":...}, {...}, {...}]  (for example, for 3 depths, 3 sentences)
        out = []
        for i in range(num_sentences):
            sentence_depths = {}
            for n, sentences in enumerate(resp_sentences):
                if i >= len(sentences):
                    sentence_depths[str(n)] = ""
                else:
                    sentence_depths[str(n)] = sentences[i]
            out.append(sentence_depths)
        
        print("-"*20 + " JSON output (NOTE: This may be only a guideline.) " + "-"*20)
        print(json.dumps(out, indent=2))
        with open(outpath, "w") as f:
            json.dump(out, f)

    # Visualize the graying:
    # NOTE: In the console, we don't have access to good levels of gray, so we use colors instead
    print("\n")
    print("-"*20 + " Word relevance visualization " + "-"*20)
    for i, c in enumerate(normed_orig_para):
        print(RESET, end='')
        color = 'black'
        if char_depths[i] >= MAX_DEPTH:
            color = 'blue'
        elif char_depths[i] >= MAX_DEPTH-1:
            color = 'green'
        elif char_depths[i] >= MAX_DEPTH-2:
            color = 'yellow'
        print(colored(c, color), end='')
    
    # HTML output
    if args.html_output is True:
        print("\n")
        print("-"*20 + " HTML code for word relevance visualization (greying) " + "-"*20)
        # Finds continuous sequences characters with the same character depth:
        sequences = extract_contiguous_sequences(char_depths)
        # Produces HTML with spans wrapped around each sequence that was the same depth:
        html_code = ""
        for seq in sequences:
            start, end, depth = seq['start'], seq['end'], seq['val']
            color_id = depth if depth < len(HTML_GRAY_LEVELS) else (len(HTML_GRAY_LEVELS)-1)
            html_code += f'<span style="color:{HTML_GRAY_LEVELS[color_id]}">' + normed_orig_para[start:end] + '</span>'
        print(html_code)
    
    # LaTeX output
    if args.latex_output is True:
        print("\n")
        print("-"*20 + " LaTeX code for word relevance visualization (greying) " + "-"*20)
        # Finds continuous sequences characters with the same character depth:
        sequences = extract_contiguous_sequences(char_depths)
        # Produces LaTeX with \textcolor{}{} wrapped around each sequence that was the same depth:
        latex_code = ""
        for seq in sequences:
            start, end, depth = seq['start'], seq['end'], seq['val']
            color_id = depth if depth < len(LATEX_GRAY_CODES) else (len(LATEX_GRAY_CODES)-1)
            if depth == 0:
                latex_code += normed_orig_para[start:end]
            else:
                latex_code += '\\textcolor{' + LATEX_GRAY_CODES[color_id] + '}{' + normed_orig_para[start:end] + '}'
        print(latex_code)