# chatgpt-extractive-shortener
Shortens a paragraph of text with ChatGPT, using successive rounds of word-level extractive summarization.

This Python script takes a paragraph as input, and runs it successively through ChatGPT to extract a shortened version that performs "nearly-extractive" summarization: a version with deleted words/phrases that don't contribute to the overall meaning, and minimal rewordings or additions.

You can use this to help edit your writing. 

## Example 

For instance, here is a quote from Stephen King's *On Writing*:

> When I talk about pictures in my mind I am talking, quite specifically, about images that shimmer around the edges. There used to be an illustration in every elementary psychology book showing a cat drawn by a patient in varying stages of schizophrenia. This cat had a shimmer around it. You could see the molecular structure breaking down at the very edges of the cat: the cat became the background and the background the cat, everything interacting, exchanging ions. People on hallucinogens describe the same perception of objects. I’m not a schizophrenic, nor do I take hallucinogens, but certain images do shimmer for me. Look hard enough, and you can’t miss the shimmer. It’s there. You can’t think too much about these pictures that shimmer. You just lie low and let them develop. You stay quiet. You don’t talk to many people and you keep your nervous system from shorting out and you try to locate the cat in the shimmer, the grammar in the picture.

We feed this through `chatgpt_extractive_shortener.py` in the command line, and get the response (non-deterministic, quality varies):

<img width="1100" alt="Screen Shot 2023-03-24 at 11 16 27 AM" src="https://user-images.githubusercontent.com/5251713/227566614-d26936e2-755b-4b7f-9915-69f9050111b8.png">

The highlighted words are ones ChatGPT wanted to delete, as they didn't contribute much to the overall meaning. Colors refer to which round of shortening the word was cut/modified at:
 - **blue** are cut first,
 - **green** are cut second,
 - **yellow** are cut in the third round or later

## Command line usage

For simple usage, you can give it a paragraph:

<img width="920" alt="Screen Shot 2023-03-24 at 11 14 56 AM" src="https://user-images.githubusercontent.com/5251713/227566102-7d1712a7-3fdd-4f32-a8f9-e0d71f1598ad.png">

Quality of responses varies widely, and the 'best' response at each level is chosen by a (fairly naive) heuristic. 

Use the `--interactive` flag if you want to 'steer' which response is chosen at each level, instead of the automatic method. 

For more parameters, run `python chatgpt_extractive_shortener.py --help`.

## Shortenings at each round

The script will also output the chosen shortenings at each successive round ('depth'), alongside a diff that visualizes the changes. For the text above (deleted words in red):

<img width="731" alt="Screen Shot 2023-03-24 at 11 16 51 AM" src="https://user-images.githubusercontent.com/5251713/227566901-b4a7050f-b667-435a-a735-c472282a9e05.png">

## What prompt do you use?
Under the hood, we use the prompt template:

> Delete 10 words or phrases from the following paragraph that don't contribute much to its meaning, but keep readability:
> "${paragraph}"
> 
> Please do not add any new words or change words, only delete words.

This prompt seems to work pretty well for ChatGPT3.5 and GPT4. GPT3.5 and 4 tend to want to reword things, so it is difficult to ask them to purely delete words. Feel free to edit the script to try your own 'shortening' prompt that might perform better for your specific needs.
