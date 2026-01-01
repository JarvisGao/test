1. Collect content and do some pre-processing
2. Use LLM to read content and generate QA pairs
3. Use LLM to augment QA pairs to cover more scenarios


In the preprocessing stage, text processing methods can be adjusted based on different file extensions. 
I use an LLM to read the files and extract corresponding QA pairs. 
I plan to utilize instruction tuning to construct these pairs for training. 
To ensure broader coverage across various scenarios, data augmentation techniques, such as synonymous substitution and learning by analogy, can be applied to the input part.
