import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pingouin import partial_corr
from scipy.stats import pearsonr, spearmanr
import matplotlib
from statannotations.Annotator import Annotator



def compute_partial(brain_vec, dnn1, dnn2, dnn3):
    df = pd.DataFrame({
        'brain': brain_vec,
        'dnn1': dnn1,
        'dnn2': dnn2,
        'dnn3': dnn3
    })
    # Compute the partial correlation between brain and dnn1, controlling for dnn2 and dnn3
    return partial_corr(data=df, x='brain', y='dnn1', covar=['dnn2', 'dnn3'], method='pearson')

def fmri_to_group(roi):
    if roi in ['V1d', 'V1v', 'V2d', 'V2v', 'V3d', 'V3v']:
        return 'Early'
    elif roi in ['IPS0', 'IPS1', 'IPS2', 'IPS3', 'IPS4', 'IPS5', 'SPL1', 'FEF', 'V3a', 'V3b', 'MST', 'hMT']:
        return 'Dorsal'
    elif roi in ['LO1', 'LO2', 'VO1', 'VO2', 'PHC1', 'PHC2', 'hV4']:
        return 'Ventral'
    else:
        return 'Other'


def fmri_plot(fmri_matrices, dnn_image_vec, dnn_semantic_vec, dnn_visual_vec):

    groups = ['Early', 'Dorsal', 'Ventral', 'Other']
    representations_labels = ['Images', 'Visual Text', 'Abstract Text']
    
    roi_labels = {group:{rep:[] for rep in representations_labels} for group in groups}

    for keys, brain_vec in fmri_matrices.items():
        partials = {
            'Images': compute_partial(brain_vec, dnn_image_vec, dnn_visual_vec, dnn_semantic_vec),
            'Visual Text': compute_partial(brain_vec, dnn_visual_vec, dnn_image_vec, dnn_semantic_vec),
            'Abstract Text': compute_partial(brain_vec, dnn_semantic_vec, dnn_image_vec, dnn_visual_vec)
        }
        for label, dnn_vec in zip(representations_labels, [dnn_image_vec, dnn_visual_vec, dnn_semantic_vec]):
            r, p = pearsonr(brain_vec, dnn_vec)
            roi_labels[fmri_to_group(keys[1])][label].append({
                "subj": keys[0], "roi": keys[1], "Representation": label,
                "Group": fmri_to_group(keys[1]), "Zero": r, 
                "Partial": partials[label]['r'].values[0]
            })
            
    base_dir = "fmri_correlation_scores"
    plots_dir = "fmri_plots"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    summary_list = []
    for group in groups:
        print(f"taking care of=== {group}")
        group_data = [item for rep in representations_labels for item in roi_labels[group][rep]]
        
        if not group_data:
            print(f"Skipping group {group} — no data.")
            continue
            
        df_group = pd.DataFrame(group_data)
        
        # שמירת הנתונים הגולמיים
        for rep in representations_labels:
            df_group_rep = df_group[df_group['Representation'] == rep]
            if not df_group_rep.empty:
                raw_path = os.path.join(base_dir, f"{group}_{rep}_raw_data.csv".replace(" ", "_"))
                df_group_rep.to_csv(raw_path, index=False)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        comparison_pairs = [
            ("Images", "Visual Text"),
            ("Images", "Abstract Text"),
            ("Visual Text", "Abstract Text")
        ]
        
        # ==================== שינוי 1: הגדרת צבעים מותאמים אישית ====================
        # הגדרת פלטת צבעים ספציפית שתואמת לתמונה שסיפקת
        custom_palette = {
            "Images": "#2c2c8d",          # כחול כהה
            "Visual Text": "#8e44ad",     # סגול
            "Abstract Text": "#b33939"   # אדום כהה
        }
        # ===========================================================================

        for i, corr_type in enumerate(['Zero', 'Partial']):
            df_summary = df_group.groupby('Representation')[corr_type].agg(['mean', 'sem']).reindex(representations_labels).reset_index()
            df_summary['Group'] = group
            df_summary['Correlation Type'] = corr_type
            summary_list.append(df_summary)

            ax = axes[i]
            
            # ======================= שינוי 2: התאמות בעיצוב הגרף =======================
            sns.barplot(
                data=df_group,
                x='Representation',
                y=corr_type,
                order=representations_labels,
                palette=custom_palette, # שימוש בפלטת הצבעים החדשה
                capsize=0.05,          # גודל קו השגיאה
                errwidth=1.5,          # עובי קו השגיאה
                ax=ax,
                edgecolor='black',     # הוספת קו מתאר שחור לעמודות
                linewidth=1.5          # עובי קו המתאר
            )
            # ===========================================================================
            
            # ======================= שינוי 3: מיקום וסגנון הטקסט =======================
            # שינוי הלולאה כדי למקם טקסט לבן ועבה במרכז כל עמודה
            for bar in ax.patches:
                height = bar.get_height()
                # הצבת הטקסט באמצע הגובה של העמודה
                ax.annotate(f'{height:.2f}', 
                            (bar.get_x() + bar.get_width() / 2., height / 2),
                            ha='center', va='center', fontsize=16, color='white', fontweight='bold')
            # ===========================================================================

            # הוספת הערות מובהקות סטטיסטית
            annotator = Annotator(ax, pairs=comparison_pairs, data=df_group, x='Representation', y=corr_type, order=representations_labels)
            annotator.configure(test='t-test_paired', text_format='star', loc='outside', verbose=0, line_width=1.5)
            annotator.apply_and_annotate()
            
            # ======================= שינוי 4: עיצוב הצירים והרקע =======================
            ax.set_title(f'{corr_type} Correlation', fontsize=14, fontweight='bold')
            ax.set_ylabel('Correlation' if i == 0 else '', fontsize=12)
            ax.set_xlabel('')
            
            # <<< הדרישה העיקרית שלך: שינוי גבולות ציר ה-Y >>>
            ax.set_ylim(0, 0.2)
            
            # הסרת הקווים העליונים והימניים של המסגרת למראה נקי
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='x', labelsize=12)
            ax.tick_params(axis='y', labelsize=11)
            # ===========================================================================
        
        plt.suptitle(f'{group} — Correlation Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        fname = f"{group}_correlation_comparison_styled.png".replace(" ", "_")
        plt.savefig(os.path.join(plots_dir, fname), dpi=300) # הגדלת הרזולוציה
        plt.close()

    summary_df = pd.concat(summary_list, ignore_index=True)
    summary_all_path = os.path.join(base_dir, "correlation_summary_all_groups.csv")
    summary_df.to_csv(summary_all_path, index=False)