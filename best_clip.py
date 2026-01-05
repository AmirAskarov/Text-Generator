
import os
import json

iterations_folder = './results/iterations'


self_ref_words = {"i", "i'm", "me", "my", "mine", "myself"}

def contains_self_reference(text):
    text_lower = text.lower()
    return any(f" {word} " in f" {text_lower} " for word in self_ref_words)

def save_best_visual():
    best_descriptions = {}
    sgpt_min = {}
    sgpt_max = {}
    all_image_ids = set()

    for filename in os.listdir(iterations_folder):
        if filename.startswith("iteration_") and filename.endswith(".json"):
            filepath = os.path.join(iterations_folder, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for image_id, info in data.items():
                    all_image_ids.add(image_id)
                    sgpt = info.get("sgpt_score", 0)
                    if image_id not in sgpt_min or sgpt < sgpt_min[image_id]:
                        sgpt_min[image_id] = sgpt
                    if image_id not in sgpt_max or sgpt > sgpt_max[image_id]:
                        sgpt_max[image_id] = sgpt

    for filename in os.listdir(iterations_folder):
        if filename.startswith("iteration_") and filename.endswith(".json"):
            filepath = os.path.join(iterations_folder, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for image_id, info in data.items():
                    clip = info.get("clip_score", 0)
                    sgpt = info.get("sgpt_score", 0)
                    sentence = info.get("D_v", "")

                    if contains_self_reference(sentence):
                        continue

                    if image_id not in best_descriptions or clip > best_descriptions[image_id]['clip_score']:
                        if sgpt == sgpt_max[image_id]:
                            sgpt_status = "max"
                        elif sgpt == sgpt_min[image_id]:
                            sgpt_status = "min"
                        else:
                            sgpt_status = None

                        best_descriptions[image_id] = {
                            'D_v': sentence,
                            'clip_score': clip,
                            'sgpt_extreme': sgpt_status
                        }

    output = {
        img_id: best_descriptions[img_id]
        for img_id in sorted(best_descriptions.keys(), key=lambda x: int(x))
    }

    with open('best_descriptions.json', 'w', encoding='utf-8') as out_f:
        json.dump(output, out_f, indent=2, ensure_ascii=False)
    excluded_images = {x for x in all_image_ids if x not in best_descriptions}
    
    return excluded_images

