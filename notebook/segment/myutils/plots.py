import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def show_cm(cm):
    return sns.heatmap(cm, annot=True, 
    annot_kws={'size':18, "color":'black'}, 
    cmap='Reds', fmt='.0f',
    square=True, cbar=False).get_figure()

def show_feature_importance(clf, X_train):
    # Extract feature names and importances
    feature_names = list(X_train.columns)
    importances = clf.get_feature_importances()
    indices = np.argsort(importances)[::-1]  # Sort features by importance

    # Create the figure and axis objects
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create a horizontal bar plot
    ax.barh(range(X_train.shape[1]), importances[indices], align='center')
    ax.set_yticks(range(X_train.shape[1]))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title('Feature Importance from Decision Tree')

    # Invert y-axis to display the most important feature at the top
    ax.invert_yaxis()

    # Return the figure object
    return fig