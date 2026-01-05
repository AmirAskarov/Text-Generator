# Multimodal Text Generation and Representation
## Project Overview
This project explores different computational representations of visual and semantic information by constructing and comparing three types of model-based representations: visual, visual-text, and abstract-text.

Visual representations were extracted from images using the CLIP model, while textual representations were generated and embedded using SGPT. For the visual-text condition, an iterative optimization process was used to generate perceptually grounded textual descriptions that align with image content while remaining semantically distinct from abstract descriptions. For each representation type, representational dissimilarity matrices (RDMs) were constructed using cosine distance, enabling a direct comparison between visual, multimodal, and abstract semantic spaces.


## How to Run the Project
An API key is required to run the text generation component.
1. Set the API key as an environment variable or replace the placeholder in `txt_generator.py`.
2. Run `txt_generator.py` to generate visual textual descriptions.
3. Run `best_clip.py` to select the descriptions with the highest CLIP score across iterations.
4. Run `best_brain_corr.py` to compute correlations between model representations and brain data.
