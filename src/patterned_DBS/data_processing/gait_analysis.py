""" GAIT analysis cDBS vs. burstDBS"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import scipy.stats as stats

from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests


from ..utils import find_folders as find_folders
from ..utils import io as io


GROUP_RESULTS_PATH = find_folders.get_patterned_dbs_project_path(folder="GroupResults")
GROUP_FIGURES_PATH = find_folders.get_patterned_dbs_project_path(folder="GroupFigures")

STIMULATION_DICT = {
    "burst": "B",
    "continuous": "A",
}


def load_gait_task_results(
    sub: str,
    stimulation: str,
    task: str,
):
    """
    Input:
        - sub: "089"
        - stimulation: "burst" or "continuous"
        - task: "10m_walk", "timed_up_and_go"

    """

    score_sheet = io.load_metadata_excel(sub=sub, sheet_name="gait_tasks")

    # only keep stimulation burst or continuous
    stim_sheet = score_sheet[
        score_sheet["stimulation"].str.contains(
            f"StimOn{STIMULATION_DICT[stimulation]}|StimOff{STIMULATION_DICT[stimulation]}",
            na=False,
        )
    ]

    # only keep the desired subscore
    subscore_data = stim_sheet.loc[stim_sheet["task"] == task]

    # get the desired scores for each sessions

    return subscore_data, stim_sheet, score_sheet


def group_gait_task_results(sub_list: list, stimulation: str, task: str):
    """
    Load the gait tasks for a list of subjects and stimulation type
    """

    # Concatenate the dataframes
    for sub in sub_list:
        subscore_data, stim_sheet, score_sheet = load_gait_task_results(
            sub=sub, stimulation=stimulation, task=task
        )
        if sub == sub_list[0]:
            all_scores = subscore_data
            all_scores["subject"] = sub
        else:
            subscore_data["subject"] = sub
            all_scores = pd.concat([all_scores, subscore_data])

    return all_scores


# def convert_time_to_float(time_str):
#     """
#     Convert a time string in the format 'HH:MM:SS' or 'HH:MM:SS min' to float (minutes).

#     Parameters:
#         time_str (str): Time in the format "HH:MM:SS" or "HH:MM:SS min".

#     Returns:
#         float: Time converted to minutes.
#     """
#     # Remove 'min' if it exists
#     time_str = time_str.replace(" min", "")

#     # Split the time string into hours, minutes, and seconds
#     try:
#         h, m, s = map(int, time_str.split(":"))
#         total_minutes = h * 60 + m + s / 60  # Convert to float minutes
#         return round(total_minutes, 2)  # Rounding to 2 decimal places
#     except ValueError:
#         print(f"Error: Incorrect time format: {time_str}")
#         return None  # Handle potential incorrect formats


def convert_time_to_seconds(time_str):
    """
    Convert a time string in the format 'Min:Sec:Milliseconds' or 'Min:Sec:Milliseconds min' to float (seconds).

    Parameters:
        time_str (str): Time in the format "Min:Sec:Milliseconds" or "Min:Sec:Milliseconds min".

    Returns:
        float: Time converted to seconds.
    """
    # Remove 'min' if it exists
    time_str = time_str.replace("min", "").strip()

    # Split the time string into minutes, seconds, and milliseconds
    try:
        m, s, ms = map(int, time_str.split(":"))
        total_seconds = (m * 60) + s + (ms / 1000)  # Convert to float seconds
        return round(total_seconds, 3)  # Round to 3 decimal places for milliseconds
    except ValueError:
        print(f"Error: Incorrect time format: {time_str}")
        return None  # Handle potential incorrect formats


def extract_data_for_bar_and_line_plot(task: str, subscore_column: str):
    """

    Extract the data for the bar and line plots for the group analysis

    Input:
        - task: "10m_walk", "timed_up_and_go"
        - subscore_column: "time" or "steps"

    Extract the scores for all subjects for
        - continuous DBS ON
        - burst DBS ON
        - StimOFF after cDBS (last StimOFF score, run 1, 2, or 3)

    Output:
        - structured_data: pd.DataFrame with columns: subject, stimulation, score
        - StimOFF_scores: pd.DataFrame with columns: subject, StimOFFA_run
    """
    sub_list = ["084", "080", "075", "086", "087", "088", "089"]

    structured_data = {
        "subject": ["sub-1", "sub-2", "sub-3", "sub-4", "sub-5", "sub-6", "sub-7"] * 3,
        "stimulation": ["continuous"] * 7 + ["burst"] * 7 + ["StimOFF"] * 7,
        "score": [],
    }

    StimOFF_scores = {
        "subject": sub_list,
        "StimOFFA_run": [],
    }

    # first extract continuous score data per subject
    for sub in sub_list:
        subscore_data, stim_sheet, score_sheet = load_gait_task_results(
            sub=sub, stimulation="continuous", task=task
        )

        StimOnA_data = subscore_data.loc[subscore_data["stimulation"] == "StimOnA"]
        if subscore_column == "time":
            score = convert_time_to_seconds(StimOnA_data[subscore_column].values[0])

        elif subscore_column == "steps":
            score = StimOnA_data[subscore_column].values[0]

        structured_data["score"].append(score)

    # then extract burst score data per subject
    for sub in sub_list:
        subscore_data, stim_sheet, score_sheet = load_gait_task_results(
            sub=sub, stimulation="burst", task=task
        )

        StimOnB_data = subscore_data.loc[subscore_data["stimulation"] == "StimOnB"]
        if subscore_column == "time":
            score = convert_time_to_seconds(StimOnB_data[subscore_column].values[0])

        elif subscore_column == "steps":
            score = StimOnB_data[subscore_column].values[0]

        structured_data["score"].append(score)

    # then extract StimOFF score data per subject: last Stim OFF score per cDSB OFF
    for sub in sub_list:
        subscore_data, stim_sheet, score_sheet = load_gait_task_results(
            sub=sub, stimulation="continuous", task=task
        )

        # check if StimOFFA_run-3 exists
        if 3 in subscore_data["run"].values:
            subscore_data = subscore_data.loc[subscore_data["run"] == 3]
            StimOFF_scores["StimOFFA_run"].append("run-3")

        elif 2 in subscore_data["run"].values:
            subscore_data = subscore_data.loc[subscore_data["run"] == 2]
            StimOFF_scores["StimOFFA_run"].append("run-2")

        elif 1 in subscore_data["run"].values:
            subscore_data = subscore_data.loc[subscore_data["run"] == 1]
            StimOFF_scores["StimOFFA_run"].append("run-1")

        else:
            print(
                "StimOffA_run-3, StimOffA_run-2, StimOffA_run-1 not found in the data"
            )
            StimOFF_scores["StimOFFA_run"].extend("not found")

        if subscore_column == "time":
            score = convert_time_to_seconds(subscore_data[subscore_column].values[0])

        elif subscore_column == "steps":
            score = subscore_data[subscore_column].values[0]

        structured_data["score"].append(score)

    return pd.DataFrame(structured_data), pd.DataFrame(StimOFF_scores)


#### PLOT WITH STATISTICS ####
# Add significance stars
def add_significance_bracket(p, start_pos, end_pos, y_max, ax, text_offset=0.1):
    """Add significance annotation to plot."""
    if p < 0.0001:
        significance = "****"
    elif p < 0.001:
        significance = "***"
    elif p < 0.01:
        significance = "**"
    elif p < 0.05:
        significance = "*"
    else:
        significance = "n.s."  # Not significant

    x_pos = (start_pos + end_pos) / 2
    ax.plot(
        [start_pos, start_pos, end_pos, end_pos],
        [y_max, y_max + text_offset, y_max + text_offset, y_max],
        lw=1.5,
        color="black",
    )
    ax.text(
        x_pos,
        y_max + text_offset * 1.5,
        significance,
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )


def plot_bar_line_group_plot_with_statistics(task: str, subscore_column: str):
    """
    Plot the bar and line plot for the group gait analysis, including statistical analysis with significance stars.

    Input:
        - task: "10m_walk", "timed_up_and_go"
        - subscore_column: "time" or "steps"

    """

    # Subject list in order
    sub_list = ["084", "080", "075", "086", "087", "088", "089"]

    # Load the structured data
    structured_data, StimOFF_scores = extract_data_for_bar_and_line_plot(
        task=task, subscore_column=subscore_column
    )

    # Statistical analysis
    pivot_data = structured_data.pivot(
        index="subject", columns="stimulation", values="score"
    )

    # Check normality for each condition
    normality_pvalues = {
        col: stats.shapiro(pivot_data[col])[1] for col in pivot_data.columns
    }
    print("Normality p-values:", normality_pvalues)

    # Determine if data is normal
    if all(p > 0.05 for p in normality_pvalues.values()):
        print("Data appears normal, performing repeated measures ANOVA...")

        # Run repeated measures ANOVA using AnovaRM
        anova = AnovaRM(
            structured_data, "score", "subject", within=["stimulation"]
        ).fit()
        print("\nANOVA Results:\n", anova.anova_table)

        p_value_anova = anova.anova_table["Pr > F"]["stimulation"]

        if p_value_anova < 0.05:
            print(
                "Significant effect found, performing paired t-tests with corrections..."
            )

            # Conduct paired t-tests
            comparisons = [
                ("continuous", "burst"),
                ("continuous", "StimOFF"),
                ("burst", "StimOFF"),
            ]
            p_values = []
            for cond1, cond2 in comparisons:
                scores1 = pivot_data[cond1]
                scores2 = pivot_data[cond2]
                _, p_val = stats.ttest_rel(scores1, scores2)
                p_values.append(p_val)

            # Multiple comparisons correction
            _, holm_corrected_pvals, _, _ = multipletests(p_values, method="holm")
            corrected_pvals = dict(zip(comparisons, holm_corrected_pvals))

    else:
        print("Data is not normal, performing Friedman test...")
        friedman_stat, friedman_p = stats.friedmanchisquare(
            pivot_data["continuous"], pivot_data["burst"], pivot_data["StimOFF"]
        )
        print(
            f"Friedman test statistic: {friedman_stat:.3f}, p-value: {friedman_p:.5f}"
        )

        if friedman_p < 0.05:
            print(
                "Significant effect found, performing Wilcoxon signed-rank tests with corrections..."
            )

            # Conduct Wilcoxon signed-rank tests
            comparisons = [
                ("continuous", "burst"),
                ("continuous", "StimOFF"),
                ("burst", "StimOFF"),
            ]
            p_values = []
            for cond1, cond2 in comparisons:
                scores1 = pivot_data[cond1]
                scores2 = pivot_data[cond2]
                _, p_val = stats.wilcoxon(scores1, scores2)
                p_values.append(p_val)

            # Multiple comparisons correction
            _, holm_corrected_pvals, _, _ = multipletests(p_values, method="holm")
            corrected_pvals = dict(zip(comparisons, holm_corrected_pvals))

    # Plot setup
    colors = plt.cm.Set2(np.linspace(0, 1, len(sub_list)))
    fig, ax = plt.subplots(figsize=(8, 6))

    # Bar plot
    sns.barplot(
        data=structured_data,
        x="stimulation",
        y="score",
        ci=None,
        color="lightgray",
        alpha=0.7,
        ax=ax,
    )

    # Scatter plot with lines for each subject
    for idx, sub in enumerate(structured_data["subject"].unique()):
        subj_data = structured_data[structured_data["subject"] == sub]
        ax.plot(
            subj_data["stimulation"],
            subj_data["score"],
            marker="o",
            label=sub,
            alpha=0.7,
            color=colors[idx],
        )

    # Define the y-axis maximum for placement of significance brackets
    y_max = structured_data["score"].max() + 1

    # check if corrected_pvals exists
    if "corrected_pvals" in locals():
        for (cond1, cond2), p_val in corrected_pvals.items():
            x_start = list(structured_data["stimulation"].unique()).index(cond1)
            x_end = list(structured_data["stimulation"].unique()).index(cond2)
            add_significance_bracket(p_val, x_start, x_end, y_max, ax)
            y_max += 1  # Adjust for spacing of brackets

    # Add titles and labels
    if subscore_column == "time":
        y_label = "Time (s)"
        y_lim = (0, 60)
    elif subscore_column == "steps":
        y_label = "Steps"
        y_lim = (0, 100)

    ax.set_title(f"Gait {task} task, {subscore_column}", fontsize=14)
    ax.set_xlabel("Stimulation", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_ylim(y_lim)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend(title="Subjects", bbox_to_anchor=(1.05, 1), loc="upper left")

    fig.tight_layout()

    # Save figure
    io.save_fig_png_and_svg(
        path=GROUP_FIGURES_PATH,
        filename=f"line_and_barplot_gait_group_analysis_{task}_{subscore_column}",
        figure=fig,
    )

    plt.show()


####### STATISTICAL ANALYSIS ########


def perform_paired_analysis(task: str, subscore_column: str):
    """
    Perform statistical tests on the given data.
    - Shapiro-Wilk test for normality
    - If normal: Repeated measures ANOVA using AnovaRM
    - If not normal: Friedman test
    - Paired t-tests or Wilcoxon signed-rank tests for post-hoc analysis with Holm and Bonferroni corrections
    - Return a summary DataFrame with statistical information
    """

    loaded_data, stim_off_sessions = extract_data_for_bar_and_line_plot(
        task=task, subscore_column=subscore_column
    )
    # loaded_data columns are: subject, stimulation, score, already in long format for AnovaRM

    # Pivot the data to wide format for easier paired testing
    pivot_data = loaded_data.pivot(
        index="subject", columns="stimulation", values="score"
    )

    # Calculate descriptive statistics
    descriptive_stats = (
        loaded_data.groupby("stimulation")["score"]
        .agg(mean="mean", std="std", median="median", min="min", max="max")
        .reset_index()
    )

    # Check normality for each stimulation condition
    normality_pvalues = {
        col: stats.shapiro(pivot_data[col])[1] for col in pivot_data.columns
    }
    descriptive_stats["normality_pvalue"] = descriptive_stats["stimulation"].map(
        normality_pvalues
    )
    print("Normality p-values:", normality_pvalues)

    results = []

    # Determine if data is normal
    if all(p > 0.05 for p in normality_pvalues.values()):
        print("Data appears normal, performing repeated measures ANOVA...")

        # Run repeated measures ANOVA using AnovaRM
        anova = AnovaRM(loaded_data, "score", "subject", within=["stimulation"]).fit()
        # score is the dependent variable, subject is the subject identifier, stimulation is the within-subject factor
        print("\nANOVA Results:\n", anova.anova_table)

        p_value = anova.anova_table["Pr > F"]["stimulation"]

        results.append(
            {
                "Test": "ANOVA",
                "statistic": anova.anova_table["F Value"]["stimulation"],
                "p_value": p_value,
            }
        )

        if p_value < 0.05:
            print(
                "Significant effect found, performing paired t-tests with corrections..."
            )

            # Conduct paired t-tests for each pair of conditions
            comparisons = [
                ("continuous", "burst"),
                ("continuous", "StimOFF"),
                ("burst", "StimOFF"),
            ]

            p_values = []
            for cond1, cond2 in comparisons:
                scores1 = pivot_data[cond1]
                scores2 = pivot_data[cond2]
                t_stat, p_val = stats.ttest_rel(scores1, scores2)
                results.append(
                    {
                        "Comparison": f"{cond1} vs {cond2}",
                        "Test": "Paired t-test",
                        "statistic": t_stat,
                        "p_value": p_val,
                    }
                )
                p_values.append(p_val)
                print(
                    f"Paired t-test between {cond1} and {cond2}: t-stat={t_stat:.3f}, p-value={p_val:.5f}"
                )

            # Apply multiple comparisons correction (Holm and Bonferroni)
            _, holm_corrected_pvals, _, _ = multipletests(p_values, method="holm")
            _, bonferroni_corrected_pvals, _, _ = multipletests(
                p_values, method="bonferroni"
            )

            for (cond1, cond2), holm_p, bonf_p in zip(
                comparisons, holm_corrected_pvals, bonferroni_corrected_pvals
            ):
                results.append(
                    {
                        "Comparison": f"{cond1} vs {cond2}",
                        "Test": "Holm Correction",
                        "p_value": holm_p,
                    }
                )
                results.append(
                    {
                        "Comparison": f"{cond1} vs {cond2}",
                        "Test": "Bonferroni Correction",
                        "p_value": bonf_p,
                    }
                )

            print("\nCorrected p-values:")
            for (cond1, cond2), holm_p, bonf_p in zip(
                comparisons, holm_corrected_pvals, bonferroni_corrected_pvals
            ):
                print(
                    f"{cond1} vs {cond2}: Holm = {holm_p:.5f}, Bonferroni = {bonf_p:.5f}"
                )

    else:
        print("Data is not normal, performing Friedman test...")

        # Perform Friedman test for repeated measures
        friedman_stat, friedman_p = stats.friedmanchisquare(
            pivot_data["continuous"], pivot_data["burst"], pivot_data["StimOFF"]
        )
        print(
            f"Friedman test statistic: {friedman_stat:.3f}, p-value: {friedman_p:.5f}"
        )

        results.append(
            {"Test": "Friedman", "statistic": friedman_stat, "p_value": friedman_p}
        )

        if friedman_p < 0.05:
            print(
                "Significant effect found, performing Wilcoxon signed-rank tests with corrections..."
            )

            # Conduct Wilcoxon signed-rank tests for each pair of conditions
            comparisons = [
                ("continuous", "burst"),
                ("continuous", "StimOFF"),
                ("burst", "StimOFF"),
            ]

            p_values = []
            for cond1, cond2 in comparisons:
                scores1 = pivot_data[cond1]
                scores2 = pivot_data[cond2]
                stat, p_val = stats.wilcoxon(scores1, scores2)
                results.append(
                    {
                        "Comparison": f"{cond1} vs {cond2}",
                        "Test": "Wilcoxon",
                        "statistic": stat,
                        "p_value": p_val,
                    }
                )
                p_values.append(p_val)
                print(
                    f"Wilcoxon signed-rank test between {cond1} and {cond2}: statistic={stat:.3f}, p-value={p_val:.5f}"
                )

            # Apply multiple comparisons correction (Holm and Bonferroni)
            _, holm_corrected_pvals, _, _ = multipletests(p_values, method="holm")
            _, bonferroni_corrected_pvals, _, _ = multipletests(
                p_values, method="bonferroni"
            )

            for (cond1, cond2), holm_p, bonf_p in zip(
                comparisons, holm_corrected_pvals, bonferroni_corrected_pvals
            ):
                results.append(
                    {
                        "Comparison": f"{cond1} vs {cond2}",
                        "Test": "Holm Correction",
                        "p_value": holm_p,
                    }
                )
                results.append(
                    {
                        "Comparison": f"{cond1} vs {cond2}",
                        "Test": "Bonferroni Correction",
                        "p_value": bonf_p,
                    }
                )

            print("\nCorrected p-values:")
            for (cond1, cond2), holm_p, bonf_p in zip(
                comparisons, holm_corrected_pvals, bonferroni_corrected_pvals
            ):
                print(
                    f"{cond1} vs {cond2}: Holm = {holm_p:.5f}, Bonferroni = {bonf_p:.5f}"
                )

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Concatenate descriptive statistics and test results vertically
    summary_df = pd.concat([descriptive_stats, results_df], axis=0, ignore_index=True)

    return summary_df, stim_off_sessions
