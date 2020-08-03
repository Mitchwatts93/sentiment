from tqdm import tqdm as tqdm
import numpy as np
import scipy

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
fig = plt.gcf()
figsize = fig.get_size_inches()
fig.set_size_inches(figsize[0]*3, figsize[1]*2) # make it wider than default - a bit of hacky solution

class Explainer():

    def __init__(self, model):
        self.model = model
        self.classes = np.array([1])#, 2, 3, 4, 5])

    def score(self, text):
        return self.model.polarity_scores(text)['compound']

    def predict(self, texts):
        probs = []
        for text in texts:
            # First, offset the float score from the range [-1, 1] to a range [0, 1]
            offset = (self.score(text) + 1) / 2.
            # Convert float score in [0, 1] to an integer value in the range [1, 5]
            #binned = np.digitize(1 * offset, self.classes) + 1
            # Similate probabilities of each class based on a normal distribution
            #simulated_probs = scipy.stats.norm.pdf(self.classes, binned, scale=0.5)
            #probs.append(simulated_probs)
            

            #score = self.score(text)
            offset = offset
            probs.append([offset])


        return np.array(probs)

def explainer(model, text):
    """Run LIME explainer on provided classifier"""
    from lime.lime_text import LimeTextExplainer

    model = Explainer(model)

    explainer = LimeTextExplainer(
        split_expression=lambda x: x.split(),
        bow=False,
        class_names=["positive probability"]
    )

    exp = explainer.explain_instance(
        text,
        num_features=20,
        top_labels=1,
        classifier_fn=model.predict,
        num_samples=5000
    )
    return exp

def visualise_sentiments(data):
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import seaborn as sns
    import pandas as pd
    cmap = cm.coolwarm.reversed()
    ax = sns.heatmap(pd.DataFrame(data).set_index("Sentence").T, center=0, annot=True, cmap=cmap)
    fig = ax.get_figure()
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(0.3)
    plt.tight_layout()
    del ax
    return fig


def return_html(whole_sentiment, whole_confidence, model, eval_text):
    lime_output = explainer(model, eval_text)
    scores = process_lime_output(lime_output)
    html = plot_findings(scores, whole_sentiment, whole_confidence)
    return html

def process_lime_output(lime_output):
    keys = [key[0] for key in lime_output.as_map()[0]]
    
    exp_list = lime_output.as_list(label=0) #this should be a list of (word,score) pairs
    zipped_words = zip(list(exp_list), keys)
    scores = [x for x, _ in sorted(zipped_words, key=lambda pair: pair[1])] # sort by the key so now in original order
    return scores

def plot_findings(scores, whole_sentiment, whole_confidence):
    import matplotlib.pyplot as plt
    fig = visualise_sentiments({
        "Sentence": ["SENTENCE"] + [score[0] for score in scores],
        "Sentiment": [whole_confidence * whole_sentiment] + [score[1] for score in scores],
    })
    html = html_from_fig(fig)
    plt.clf()
    del fig
    del plt
    return html

def html_from_fig(fig):
    from io import BytesIO
    import base64
    buf = BytesIO()
    fig.savefig(buf, format="png", )
    # Embed the result in the html output.
    data = base64.b64encode(buf.getbuffer()).decode("ascii") # see img scaling setup for html?
    return f"<img src='data:image/png;base64,{data}'; style='height: 100%; width: 100%; object-fit: contain'/>" # f"<img src='data:image/png;base64,{data}'/>"
