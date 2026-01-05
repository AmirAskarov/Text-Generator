import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pingouin import partial_corr
from scipy.stats import pearsonr, sem
import matplotlib



def compute_partial(brain_vec, dnn1, dnn2, dnn3):
    df = pd.DataFrame({
        'brain': brain_vec,
        'dnn1': dnn1,
        'dnn2': dnn2,
        'dnn3': dnn3
    })
    # Compute the partial correlation between brain and dnn1, controlling for dnn2 and dnn3
    return partial_corr(data=df, x='brain', y='dnn1', covar=['dnn2', 'dnn3'], method='pearson')

def meg_plot(meg_matrices, dnn_image_vec, dnn_semantic_vec, dnn_visual_vec):

    
    
    session = [{
    'Images': [],
    'Visual Text': [],
    'Abstract Text': []
    } for _ in range(2)]
    
    for keys, brain_vec in meg_matrices.items():
            
            # Partial correlations
            partials= {
            'Images':compute_partial(brain_vec, dnn_image_vec, dnn_visual_vec, dnn_semantic_vec),
            'Visual Text': compute_partial(brain_vec, dnn_visual_vec, dnn_image_vec, dnn_semantic_vec),
            'Abstract Text':compute_partial(brain_vec, dnn_semantic_vec, dnn_image_vec, dnn_visual_vec)
            }
            # Zero-order correlations
            for label, dnn_vec in zip(['Images', 'Visual Text', 'Abstract Text'],
                                    [dnn_image_vec, dnn_visual_vec, dnn_semantic_vec]):
                r, p = pearsonr(brain_vec, dnn_vec)
                session[keys[1] - 1][label].append({"subj": keys[0], "ms": keys[2],
                 "Representation": label,
                 "Zero": r,
                 "Partial": partials[label]['r'].values[0]})
    
    results_base = "meg_correlation_results"
    plots_dir = "meg_correlation_plots"
    os.makedirs(results_base, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    for i, sess in enumerate(session):  
        session_name = f"session_{i+1}"
        session_dir = os.path.join(results_base, session_name)
        os.makedirs(session_dir, exist_ok=True)

        all_data = []

        for label in ['Images', 'Visual Text', 'Abstract Text']:
            rep_data = []
            for entry in sess[label]:
                row = {
                    'subj': entry['subj'],
                    'ms': entry['ms'],
                    'Representation': label,
                    'Zero': entry['Zero'],
                    'Partial': entry['Partial']
                }
                rep_data.append(row)
                all_data.append(row)

            df_rep = pd.DataFrame(rep_data)
            csv_path = os.path.join(session_dir, f"{label.replace(' ', '_')}.csv")
            df_rep.to_csv(csv_path, index=False)

        df_all = pd.DataFrame(all_data)
        df_all = df_all.sort_values(by='ms')
        df_summary = df_all.groupby(['ms', 'Representation']).agg(
            Zero_mean=('Zero', 'mean'),
            Zero_std=('Zero', 'std'),
            Zero_sem=('Zero', lambda x: sem(x, nan_policy='omit')),
            Partial_mean=('Partial', 'mean'),
            Partial_std=('Partial', 'std'),
            Partial_sem=('Partial', lambda x: sem(x, nan_policy='omit'))
        ).reset_index()

        summary_csv_path = os.path.join(session_dir, f"session_{i+1}_summary.csv")
        df_summary.to_csv(summary_csv_path, index=False)
        
        plt.figure(figsize=(10, 6))
        for label in ['Images', 'Visual Text', 'Abstract Text']:
            data_label = df_summary[df_summary['Representation'] == label]
            plt.plot(data_label['ms'], data_label['Zero_mean'], label=label)
            # הוספת SEM כאזור בצבע חצי שקוף
            plt.fill_between(
                data_label['ms'],
                data_label['Zero_mean'] - data_label['Zero_sem'],
                data_label['Zero_mean'] + data_label['Zero_sem'],
                alpha=0.2
            )

        plt.axvline(0, color='gray', linestyle='--', label='stimulus onset')
        plt.title(f"Session {i+1} - Zero-order correlations with SEM")
        plt.xlabel("Time (ms)")
        plt.ylabel("Correlation")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plot_path = os.path.join(plots_dir, f"session_{i+1}_zero_corr_sem.png")
        plt.savefig(plot_path)
        plt.close()

        # ===================== גרף Partial correlations עם SEM =====================
        plt.figure(figsize=(10, 6))
        for label in ['Images', 'Visual Text', 'Abstract Text']:
            data_label = df_summary[df_summary['Representation'] == label]
            plt.plot(data_label['ms'], data_label['Partial_mean'], label=label)
            # הוספת SEM Partial
            plt.fill_between(
                data_label['ms'],
                data_label['Partial_mean'] - data_label['Partial_sem'],
                data_label['Partial_mean'] + data_label['Partial_sem'],
                alpha=0.2
            )

        plt.axvline(0, color='gray', linestyle='--', label='stimulus onset')
        plt.title(f"Session {i+1} - Partial correlations with SEM")
        plt.xlabel("Time (ms)")
        plt.ylabel("Partial Correlation")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plot_path = os.path.join(plots_dir, f"session_{i+1}_partial_corr_sem.png")
        plt.savefig(plot_path)
        plt.close()