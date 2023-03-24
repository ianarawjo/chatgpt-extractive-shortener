from difflib import SequenceMatcher
from termcolor import colored
import re

JUNK_CHARS = ''.join((',', '.', ';', ':', '"', "'", '-'))

# Paragraphs for testing:
# paragraph1 = """This teaching experiment invites students into an urgent conversation about the ethics of using AI in coursework. Especially for courses that involve writing, students and teachers must necessarily confront the problem of plagiarism with the wide availability of electronic sources and online essay mills. As will be discussed below, the concept of “plagiarism” itself needs more nuance, and it certainly gets blurrier in context of using an AI. But in its familiar form, especially as defined institutional settings like my own university, plagiarism appears as the uncredited, knowing, and sometimes wholesale adaptation of work that is not one’s own. For courses involving writing, that often manifests as text copied from online sources or acquired from vendors. Interestingly, artificial intelligence has upped the ante, disabling the usual strategies of detecting plagiarism by Googling phrases or checking papers against a known database (e.g. Turnitin). Some online resources now advertise AI to students to generate usable, unique text that is untraceable by current plagiarism detection software. For example, claiming in a tagline to be “Empowered by Artificial Intelligence,” EssayBot.com promises itself as “your personal AI writing tool. With your essay title, EssayBot suggests most relevant contents. It paraphrases for you to erase plagiarism concerns.” Marketing itself as a “bot” designed to outmaneuver plagiarism, the site feeds into concerns about the automation of writing and the erasure of human effort. In response, learning management systems and plagiarism detection software are now adapting AI tools of their own, locked in an arms race between crisply defined antagonists: systems to cheat artificially versus systems to insure original work. Staking out this battle line, one recent plagiarism detection product simply calls itself “Turnitin Originality.”"""
# paragraph2 = """This teaching experiment invites students into a conversation about the ethics of using AI in coursework. Especially for courses involving writing, students and teachers must confront plagiarism with electronic sources and online essay mills. As discussed below, "plagiarism" needs nuance, and it gets blurrier using AI. Plagiarism appears as the adaptation of work that is not one’s own. AI disables detecting plagiarism by Googling phrases or checking papers against a database (e.g. Turnitin). Some resources advertise AI to generate unique text untraceable by current plagiarism detection software. EssayBot.com promises itself as “your personal AI writing tool. It paraphrases for you to erase plagiarism concerns.” The site feeds concerns about automation and erasure of human effort. Learning management systems and plagiarism detection software adapt AI tools, locked in an arms race: systems to cheat artificially versus systems to insure original work. One recent product calls itself “Turnitin Originality.”"""

"""
    Get the differences between two sequences of text.
"""
def diff_text(paragraph1: str, paragraph2: str, print_result: bool=True):

    # Split them into words by whitespace, so we diff on words instead of letters:
    p1, p2 = paragraph1.split(), paragraph2.split()

    # For each word, there can be punctuation attached. We want to remove the punctuation,
    # -- so that the diff doesn't care about it --but, we also want to keep track of the original
    # words *with* punctuation, so we can reconstruct the sentence. 
    p1_orig, p2_orig = p1[:], p2[:]
    p1 = [_cleaned(word) for word in p1]
    p2 = [_cleaned(word) for word in p2]

    # Diff on words, using difflib:
    s = SequenceMatcher(None, p1, p2)
    opcodes = s.get_opcodes()

    # Print the diff with added words in green and removed words in red
    if print_result is True:
        _print_diff_colored(p1, p2, opcodes)
    
    return opcodes

def _print_diff_colored(p1, p2, opcodes):
    formatted_paragraph = ""
    for opcode in opcodes:
        # (tag, i1, i2, j1, j2) = opcode
        # print("%s paragraph1[%d:%d] (%s) paragraph2[%d:%d] (%s)" % (tag, i1, i2, ' '.join(p1[i1:i2]), j1, j2, ' '.join(p2[j1:j2])))

        if opcode[0] == 'equal':
            # No change in this segment, just add it to both formatted paragraphs
            formatted_paragraph += ' '.join(p1[opcode[1]:opcode[2]])
        elif opcode[0] == 'delete':
            # This segment was removed in the second paragraph, color it red in the first paragraph
            formatted_paragraph += " \033[31m" + ' '.join(p1[opcode[1]:opcode[2]]) + "\033[0m "
        elif opcode[0] == 'replace':
            # This segment was replaced in the first paragraph, color it red in the first paragraph and green in the second
            formatted_paragraph += " (\033[33m" + ' '.join(p1[opcode[1]:opcode[2]]) + "\033[0m ->"
            formatted_paragraph += "\033[34m" + ' '.join(p2[opcode[3]:opcode[4]]) + "\033[0m) "
        elif opcode[0] == 'insert':
            # This segment was added in the second paragraph, color it green in the second paragraph
            formatted_paragraph += " \033[32m" + ' '.join(p2[opcode[3]:opcode[4]]) + "\033[0m "
    print(formatted_paragraph)

def _cleaned(s: str) -> str: 
    if len(s) == 0: return s
    orig_s = s
    s = s.lower()
    s = s.translate(str.maketrans('', '', JUNK_CHARS))
    return s
