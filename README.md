# Text-Generator
## Project Overview
This project explores different computational representations of visual and semantic information by constructing and comparing three types of model-based representations: visual, visual-text, and abstract-text.

Visual representations were extracted from images using the CLIP model, while textual representations were generated and embedded using SGPT. For the visual-text condition, an iterative optimization process was used to generate perceptually grounded textual descriptions that align with image content while remaining semantically distinct from abstract descriptions. For each representation type, representational dissimilarity matrices (RDMs) were constructed using cosine distance, enabling a direct comparison between visual, multimodal, and abstract semantic spaces.


## How to Run the Project
1. Start by running the txt_generator.py file to generate the visual descriptions.  
2. Next, run best_clip.py to save the visual descriptions with the best clip score from all 10 iterations.  
3. Finally, run best_brain_corr.py to calculate the correlations between all the DNNs and the brain data.
