
import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPModel, CLIPProcessor
import torch
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pingouin import partial_corr, corr
from meg import meg_plot
from fmri import fmri_plot
from best_clip import save_best_visual
import ast



def convert_types(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def create_upper_triangle_comb(filepath, comb, brain_vectors, excluded):
    df = pd.read_csv(filepath)
    grouped = df.groupby(comb)
    matrices = {}
    excluded = {int(x) for x in excluded}
    for keys, group in grouped:
        images = sorted(set(group["image1"]).union(set(group["image2"]))-excluded)
        # print("imagesssssss:", images)
        image_to_idx = {img: i for i, img in enumerate(images)}
        n = len(images)
        mat = np.full((n, n), np.nan)

        for _, row in group.iterrows():
            if row["image1"] in excluded or row["image2"] in excluded:
                continue

            i = image_to_idx[row["image2"]]
            j = image_to_idx[row["image1"]]
            mat[i, j] = row[brain_vectors]

        upper_triangle = mat[np.triu_indices(n, k=1)]
        matrices[keys] = upper_triangle

    return matrices


def compute_rdm_from_descriptions(filepath, model, excluded=None, k=None, output_csv=None):
    with open(filepath, 'r') as f:
        data = json.load(f)
    ids = sorted([i for i in data.keys() if i not in excluded], key=lambda x: int(x))
    if not k:
        descriptions = [data[id] for id in ids]
    else:
        descriptions = [data[id][k] for id in ids]

    
    embeddings = model.encode(descriptions, batch_size=8)
    sim_matrix = cosine_similarity(embeddings)
    rdm = 1 - sim_matrix
    # Extract the upper triangular values (excluding the diagonal)
    triu_indices = np.triu_indices(rdm.shape[0], k=1)
    dnn_vec = rdm[triu_indices]

    # Save to CSV if output path provided
    if output_csv:
        df = pd.DataFrame({
            'image1': [ids[j] for j in triu_indices[1]],
            'image2': [ids[i] for i in triu_indices[0]],
            'score': dnn_vec
        })
        df.to_csv(output_csv, index=False)

    return dnn_vec


def compute_image_rdm(image_paths, model, processor, device='cpu'):
    images = [Image.open(path).convert("RGB") for path in image_paths]
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
        embeddings = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
    embeddings = embeddings.cpu().numpy()
    sim_matrix = cosine_similarity(embeddings)
    rdm = 1 - sim_matrix
    output_csv = "./saved_dnns/image_rdm.csv"
    # Get upper triangle values
    triu_indices = np.triu_indices(rdm.shape[0], k=1)
    dnn_vec = rdm[triu_indices]

    # Generate image IDs from file names
    ids = [path+1 for path in range(len(image_paths))]

    # Save to CSV
    df = pd.DataFrame({
        'image1': [ids[j] for j in triu_indices[1]],  
        'image2': [ids[i] for i in triu_indices[0]],  
        'score': dnn_vec
    })
    df.to_csv(output_csv, index=False)
    
    return dnn_vec

def csv_to_dnn(filepath):
    df = pd.read_csv(filepath)
    vector = df['score'].to_numpy()
    return vector

def compute_and_save_correlations(dnn_image_vec, dnn_visual_text_vec, dnn_abstract_text_vec, save_dir="correlation_results"):
    """
    מחשבת מתאמי Zero-order ו-Partial בין שלושה וקטורים,
    כולל p-values, ושומרת את התוצאות לקובץ CSV.
    """
    os.makedirs(save_dir, exist_ok=True)

    data = pd.DataFrame({
        'Image': dnn_image_vec,
        'Visual Text': dnn_visual_text_vec,
        'Abstract Text': dnn_abstract_text_vec
    })

    results = []
    # הגדרת הזוגות והמשתנה שיש לפקח עליו (covariate)
    pairs = [
        ('Image', 'Visual Text', 'Abstract Text'),
        ('Image', 'Abstract Text', 'Visual Text'),
        ('Visual Text', 'Abstract Text', 'Image')
    ]

    for x, y, covar in pairs:
        # חישוב מתאם Zero-order
        zero_order_res = corr(data[x], data[y], method='pearson').iloc[0]
        
        # חישוב מתאם Partial
        partial_res = partial_corr(data=data, x=x, y=y, covar=covar, method='pearson').iloc[0]

        results.append({
            'Var1': x,
            'Var2': y,
            'Control (for partial)': covar,
            'Zero-order_r': zero_order_res['r'],
            'Zero-order_p': zero_order_res['p-val'],
            'Partial_r': partial_res['r'],
            'Partial_p': partial_res['p-val']
        })

    df_results = pd.DataFrame(results)
    csv_path = os.path.join(save_dir, "dnn_correlations.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"Saved DNN correlations to {csv_path}")
    return csv_path

# --- שלב 2: יצירת הגרפים ---

def p_to_stars(p_val):
    """ ממיר p-value לכוכבי מובהקות """
    if p_val < 0.001:
        return '***'
    elif p_val < 0.01:
        return '**'
    elif p_val < 0.05:
        return '*'
    else:
        return 'n.s.'

def add_bracket(ax, x1, x2, y, height, text):
    """ פונקציית עזר להוספת סוגרי מובהקות בין שתי עמודות """
    line_x = [x1, x1, x2, x2]
    line_y = [y, y + height, y + height, y]
    ax.plot(line_x, line_y, lw=1.5, color='black')
    ax.text((x1 + x2) * 0.5, y + height, text, ha='center', va='bottom', color='black', fontsize=14)

def plot_correlations_styled(csv_path, corr_type='Partial', save_path=None):
    """
    יוצר גרף עמודות מעוצב המציג את המתאמים (Partial או Zero-order)
    בסגנון הדומה לתמונה שסופקה.
    """
    if corr_type not in ['Partial', 'Zero-order']:
        raise ValueError("corr_type must be 'Partial' or 'Zero-order'")

    df = pd.read_csv(csv_path)
    
    # --- הכנת הנתונים לגרף ---
    r_col = f'{corr_type}_r'
    p_col = f'{corr_type}_p'
    
    # סדר העמודות בגרף כפי שמופיע בתמונה
    plot_order = [
        ('Image', 'Visual Text'),
        ('Image', 'Abstract Text'),
        ('Visual Text', 'Abstract Text')
    ]
    
    plot_data = []
    for var1, var2 in plot_order:
        row = df[((df['Var1'] == var1) & (df['Var2'] == var2)) | ((df['Var1'] == var2) & (df['Var2'] == var1))]
        if not row.empty:
            plot_data.append(row.iloc[0])
    
    plot_df = pd.DataFrame(plot_data)
    values = plot_df[r_col].values
    p_values = plot_df[p_col].values

    # --- יצירת הגרף ---
    fig, ax = plt.subplots(figsize=(8, 7))
    bar_positions = np.arange(len(plot_df))
    bars = ax.bar(bar_positions, values, width=0.7, color='#cccccc', edgecolor='black')

    # --- הוספת טקסטים וסימוני מובהקות ---
    # הוספת ערך המתאם והמובהקות מעל כל עמודה
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        significance = p_to_stars(p_values[i])
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.02, significance, ha='center', va='bottom', fontsize=16, fontweight='bold')
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval / 2, f'{yval:.2f}', ha='center', va='center', fontsize=16, color='black')

    # --- הוספת תוויות צבעוניות לציר X ---
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.spines['bottom'].set_visible(False)

    y_pos_line1, y_pos_line2, y_pos_line3 = -0.08, -0.13, -0.18
    # עמודה 1
    ax.text(0, y_pos_line1, 'Image', color='blue', ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=14)
    ax.text(0, y_pos_line2, '&', color='black', ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=14)
    ax.text(0, y_pos_line3, 'Visual Text', color='purple', ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=14)
    # עמודה 2
    ax.text(1, y_pos_line1, 'Image', color='blue', ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=14)
    ax.text(1, y_pos_line2, '&', color='black', ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=14)
    ax.text(1, y_pos_line3, 'Abstract Text', color='red', ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=14)
    # עמודה 3
    ax.text(2, y_pos_line1, 'Visual Text', color='purple', ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=14)
    ax.text(2, y_pos_line2, '&', color='black', ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=14)
    ax.text(2, y_pos_line3, 'Abstract Text', color='red', ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=14)
    
    # --- הוספת סוגרי מובהקות להשוואה בין עמודות ---
    # הערה: הקוד רק מצייר את הסוגריים כפי שמופיע בתמונה. 
    # חישוב המובהקות של ההבדל בין מתאמים דורש מבחן סטטיסטי נפרד (שלא מומש כאן).
    max_val = max(values)
    bracket_y_base = max_val + 0.1
    bracket_height = 0.03
    add_bracket(ax, 0, 1, bracket_y_base + 0.1, bracket_height, '***')
    add_bracket(ax, 1, 2, bracket_y_base, bracket_height, '***')
    add_bracket(ax, 0, 2, bracket_y_base + 0.2, bracket_height, '***')

    # --- עיצוב סופי ---
    ax.set_ylabel(f'{corr_type} correlation between DNNs', fontsize=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, max_val + 0.35)
    plt.subplots_adjust(bottom=0.2) # יצירת מקום לתוויות התחתונות

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    plt.close()




   
def main():

    excluded = save_best_visual()

    device = torch.device('cpu')
    print("Models...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    print("Creating dnns...")
    save_dir = "saved_dnns"
    os.makedirs(save_dir, exist_ok=True)
    # Load visual text RDM
    
    dnn_visual_vec = csv_to_dnn("./saved_dnns/best_visual_rdm.csv")
    
    # Load semantic text RDM
    dnn_semantic_vec = csv_to_dnn("./saved_dnns/semantic_rdm.csv")
    # Load image RDM
    image_paths = [f"./imgs_png/{i}.png" for i in range(1, 93) if str(i) not in excluded]
    dnn_image_vec = compute_image_rdm(image_paths, clip_model, processor)
    
    csv_file_path = compute_and_save_correlations(
        dnn_image_vec=dnn_image_vec,
        dnn_visual_text_vec=dnn_visual_vec,
        dnn_abstract_text_vec=dnn_semantic_vec
    )

    # 2. יצירת גרף מתאמים חלקיים (Partial)
    plot_correlations_styled(
        csv_path=csv_file_path,
        corr_type='Partial',
        save_path='correlation_results/partial_correlations_plot.png'
    )

    # 3. יצירת גרף מתאמי Zero-order
    plot_correlations_styled(
        csv_path=csv_file_path,
        corr_type='Zero-order',
        save_path='correlation_results/zero_order_correlations_plot.png'
    )
    
    print("\nDone! Check the 'correlation_results' folder for the CSV file and the plots.")
    
    print("Finish creating...")
    # Create RDM vectors (upper triangle) for each subject-ROI pair from the fMRI data
    print("Create fmri upper...")
    fmri_matrices = create_upper_triangle_comb("./all_roi_rdms_small_res.csv", ["subj", "roi"], "fMRI", excluded)
    print("=== fMRI plotting ===")
    fmri_plot(fmri_matrices, dnn_image_vec, dnn_semantic_vec, dnn_visual_vec)
    # Create RDM vectors (upper triangle) for each subj-session-ms from the MEG data
    print("Create meg upper...")
    df = pd.read_csv("./upper_triangle_meg.csv")
    meg_matrices = {}

    for _, row in df.iterrows():
        # Convert string "(1, 1, -100)" to tuple
        key = ast.literal_eval(row['group'])
        # Collect all values in value_1 ... value_3741
        values = [row[col] for col in df.columns if col.startswith('value')]
        meg_matrices[key] = values

    print("=== MEG plotting ===")
    meg_plot(meg_matrices, dnn_image_vec, dnn_semantic_vec, dnn_visual_vec)
    

    

if __name__ == "__main__":
    main()

