import matplotlib.pyplot as plt
import pandas as pd
import os
import json

def draw_accuracy_vs_pass_for_all_days_accuracy(
    accuracy_day_pass,
    accuracy_list_bl_weighted,
    accuracy_name='F1 Score',
    title='F1 Score vs Pass for Each Day'
):

    num_passes = len(accuracy_day_pass)
    num_days = len(accuracy_day_pass[0])

    plt.rcParams.update({
        "font.size": 34,
        "axes.labelsize": 38,
        "axes.titlesize": 42,
        "legend.fontsize": 34,
        "xtick.labelsize": 34,
        "ytick.labelsize": 34
    })


    cols = 3
    rows = (num_days + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(26, 7 * rows))
    fig.suptitle(title, fontsize=55, y=0.98)

    axes = axes.flatten()
    pass_with_direction = [f'{i}' for i in range(1, num_passes + 1)]

    colors = {
        "RL": "#1f77b4", 
        "Baseline": "#ff7f0e"
    }

    lines_for_legend = []

    for day in range(num_days):
        ax = axes[day]
        day_accuracies = [accuracy_day_pass[pass_num][day] for pass_num in range(num_passes)]
        baseline = [accuracy_list_bl_weighted[day]] * num_passes

        line_rl, = ax.plot(
            pass_with_direction,
            day_accuracies,
            marker='o',
            markersize=18,
            linestyle='-',
            color=colors["RL"],
            linewidth=5,
            label='HMM-RL'
        )

        line_bl, = ax.plot(
            pass_with_direction,
            baseline,
            marker='s',
            markersize=18,
            linestyle='--',
            color=colors["Baseline"],
            linewidth=5,
            label='Baseline'
        )

        if day == 0:
            lines_for_legend = [line_rl, line_bl]

        ax.set_title(f'Day {day + 1}')
        ax.set_xlabel('Pass')
        ax.set_ylabel(accuracy_name)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle='--', alpha=0.6)

    for ax in axes[num_days:]:
        ax.axis('off')

    fig.legend(
        handles=lines_for_legend,
        labels=[line.get_label() for line in lines_for_legend],
        loc='upper center',
        bbox_to_anchor=(0.5, 0.96),
        ncol=2,
        frameon=False,
        fontsize=35
    )

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    folder_path = 'code/data_market/output'
    file_name = f"{title}.png"
    full_path = f"{folder_path}/{file_name}"

    os.makedirs(folder_path, exist_ok=True)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')

    print(f"Graph for {accuracy_name} vs. pass for each day saved to {full_path}.")
    plt.close()

def draw_accuracy_vs_day_for_prediction_accuracy(
    accuracy_list_weighted,
    accuracy_list_bl_weighted,
    accuracy_name='F1 Score',
    title='F1 Score vs Day for Prediction'
):
    num_days = len(accuracy_list_weighted)
    days = list(range(1, num_days + 1))

    plt.rcParams.update({
        "font.size": 34,
        "axes.labelsize": 38,
        "axes.titlesize": 42,
        "legend.fontsize": 34,
        "xtick.labelsize": 34,
        "ytick.labelsize": 34
    })

    colors = {
        "RL": "#1f77b4",
        "Baseline": "#ff7f0e"
    }

    fig, ax = plt.subplots(figsize=(25, 20))
    fig.suptitle(title, fontsize=55, y=0.92)

    line_rl, = ax.plot(
        days,
        accuracy_list_weighted,
        marker='o',
        markersize=18,
        linestyle='-',
        color=colors["RL"],
        linewidth=5,
        label='HMM-RL'
    )

    line_bl, = ax.plot(
        days,
        accuracy_list_bl_weighted,
        marker='s',
        markersize=18,
        linestyle='--',
        color=colors["Baseline"],
        linewidth=5,
        label='Baseline'
    )

    ax.set_xlabel('Day')
    ax.set_ylabel(accuracy_name)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle='--', alpha=0.6)

    ax.legend(
        handles=[line_rl, line_bl],
        loc='upper center',
        bbox_to_anchor=(0.5, 1.1),
        ncol=2,
        frameon=False,
        fontsize=35
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    folder_path = 'code/data_market/output'
    file_name = f"{title}.png"
    full_path = f"{folder_path}/{file_name}"

    os.makedirs(folder_path, exist_ok=True)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"Graph for {accuracy_name} vs. day saved to {full_path}.")
    plt.close()

def compare_hyperparameters(json_data):
    data = []
    for exp in json_data['experiments']:
        data.append({
            "delta": exp['parameters']['delta'],
            "beta": exp['parameters']['beta'],
            "k": exp['parameters']['k'],
            "gamma": exp['parameters']['gamma'],
            "hmm_f1": exp['hmm_rl_result']['f1_score'],
            "hmm_f1_score_list_all_pass": exp['hmm_rl_result']['f1_score_list_all_pass'],
            "hmm_f1_score_list_last_pass": exp['hmm_rl_result']['f1_score_list_last_pass'],
            "baseline_f1": exp['baseline_result']['f1_score'],
            "baseline_f1_score_list": exp['baseline_result']['f1_score_list'],
        })

    df = pd.DataFrame(data)

    optimal = df.loc[df['hmm_f1'].idxmax()]
    optimal_params = optimal.to_dict()

    print(f"Optimal hyperparameters:\n  δ = {optimal_params['delta']}\n  β = {optimal_params['beta']}\n  k = {optimal_params['k']}\n  γ = {optimal_params['gamma']:.2f}")

    draw_accuracy_vs_pass_for_all_days_accuracy(
        optimal_params["hmm_f1_score_list_all_pass"],
        optimal_params["baseline_f1_score_list"],
        title=f'F1 Score vs. Pass for Each Day (δ = {optimal_params["delta"]})'
    )

    draw_accuracy_vs_day_for_prediction_accuracy(
        optimal_params["hmm_f1_score_list_last_pass"],
        optimal_params["baseline_f1_score_list"],
        title=f'F1 Score vs. Day for Prediction (Final Pass, δ = {optimal_params["delta"]})'
    )

    param_names = ['gamma', 'beta', 'k', 'delta']
    plot_titles = {
        'gamma': r'Effect of $\gamma$',
        'beta': r'Effect of $\beta$',
        'k': r'Effect of $k$',
        'delta': r'Effect of $\delta$'
    }

    param_symbols = {
        'gamma': r'$\gamma$',
        'beta': r'$\beta$',
        'k': r'$k$',
        'delta': r'$\delta$'
    }


    colors = {
        "RL": "#1f77b4",
        "Baseline": "#ff7f0e"
    }

    fig, axes = plt.subplots(2, 2, figsize=(26, 21))
    fig.suptitle('Effect of Hyperparameters on F1 Score', fontsize=55, y=0.975)

    for ax, param in zip(axes.flatten(), param_names):
        subset = df.copy()
        for other in param_names:
            if other != param:
                subset = subset[subset[other] == optimal[other]]

        if subset.empty:
            ax.axis('off')
            continue

        line_rl, = ax.plot(
            subset[param],
            subset['hmm_f1'],
            marker='o',
            markersize=18,
            linestyle='-',
            linewidth=5,
            color=colors["RL"],
            label='HMM-RL'
        )

        line_bl, = ax.plot(
            subset[param],
            subset['baseline_f1'],
            marker='s',
            markersize=18,
            linestyle='--',
            linewidth=5,
            color=colors["Baseline"],
            label='Baseline'
        )

        for x, y in zip(subset[param], subset['hmm_f1']):
            ax.text(x, y + 0.02, f'{y:.2f}', ha='center', fontsize=32, color=colors["RL"])

        for x, y in zip(subset[param], subset['baseline_f1']):
            ax.text(x, y + 0.02, f'{y:.2f}', ha='center', fontsize=32, color=colors["Baseline"])

        ax.set_title(plot_titles[param], pad = 20)
        ax.set_xlabel(param_symbols[param])
        ax.set_ylabel('F1 Score')
        ax.set_ylim(0.3, 0.7)
        ax.grid(True, linestyle='--', alpha=0.6)

    fig.legend(
        handles=[line_rl, line_bl],
        loc='upper center',
        bbox_to_anchor=(0.5, 0.93),
        ncol=2,
        frameon=False,
        fontsize=35
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    folder_path = 'code/data_market/output'
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, "Effect of Hyperparameters.png")
    plt.savefig(file_path, bbox_inches='tight', dpi=300)
    print(f"Saved hyperparameter comparison plot to: {file_path}")
    plt.close()


if __name__ == "__main__":
    # Load JSON data
    with open("code/data_market/output/hmm_rl_results.json") as f:
    # with open("code/data_market/output/hmm_rl_results_full.json") as f:
        data = json.load(f)

    compare_hyperparameters(data)