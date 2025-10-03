import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('/Users/apocalyvec/Downloads/Sweyepe Questionaire.csv')

# plot the TLX scores, with each dimension as a separate plot
"""
the original df's column is like this:
Participant ID  Gender  Session #   Mental Demand: how mentally demanding was the task? [FingerTap] Mental Demand: how mentally demanding was the task? [Gaze&Pinch]    Mental Demand: how mentally demanding was the task? [Swipe]   Physical Demand: how physically demanding was the task? [FingerTap]  Physical Demand: how physically demanding was the task? [Gaze&Pinch] Physical Demand: how physically demanding was the task? [Swipe]    Temporal Demand: how hurried or rushed was the pace of the task? [FingerTap]    Temporal Demand: how hurried or rushed was the pace of the task? [Gaze&Pinch]   Temporal Demand: how hurried or rushed was the pace of the task? [Swipe]    Performance: how successful were you in accomplishing what you were asked to do? [FingerTap]    Physical Demand: how physically demanding was the task? [Gaze&Pinch]    Physical Demand: how physically demanding was the task? [Swipe]

, as such, the conditions are in the column names (<tlx question> <condition>) and the scores are in the cells.
we need to melt the df to have the conditions as a column, and the scores as a column.
we should create a new melted df for each dimension in the tlx scores.
"""
condition_mapping = {'FingerTap': 'HandTap',
                     'Gaze&Pinch': 'GazePinch',
                     'Swipe': 'Sweyepe'}
cols = ['Mental Demand: how mentally demanding was the task?',
        'Physical Demand: how physically demanding was the task?',
        'Temporal Demand: how hurried or rushed was the pace of the task?',
        'Performance: how successful were you in accomplishing what you were asked to do?',
        'Effort: how hard did you have to work to accomplish your level of performance?',
        'Frustration: how insecure, discouraged, irritated, stressed, and annoyed were you?',
        ]

melted_dfs = {}
for col in cols:
    relevant_cols = [c for c in df.columns if col in c]  # Select only matching columns
    melted_df = df.melt(id_vars=['Participant ID', 'Gender', 'Session #'],
                        value_vars=relevant_cols,
                        var_name='Condition',
                        value_name='Score')

    # Extract the condition from the column name
    melted_df['Condition'] = melted_df['Condition'].apply(lambda x: condition_mapping[x.split('[')[1].split(']')[0]])
    # rename the column 'Session #' to 'Session'
    melted_df.rename(columns={'Session #': 'Session'}, inplace=True)
    # extract the number for the score
    melted_df['Score'] = melted_df['Score'].apply(lambda x: int(x if len(x) == 1 else x.split(' ')[0]))

    melted_dfs[col] = melted_df

# plot the TLX scores, with each dimension as a separate plot
for col in cols:
    g = sns.catplot(x='Session', y='Score', data=melted_dfs[col], kind='bar', hue='Condition')
    # the title is the text before the first colon
    plt.title(col.split(':')[0])
    g.fig.subplots_adjust(top=0.9)

    # plt.tight_layout()
    # put the legend to upper right
    # plt.legend(loc='upper right')
    plt.show()