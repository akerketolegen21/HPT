
from collections import deque
from networkx import DiGraph
import networkx
from sklearn_hierarchical_classification.constants import ROOT
import sys
sys.path.append('.')
hierarchy = {
    'Attack on Reputation': ['Name calling/Labeling', 'Doubt', 'Smears', 'Reductio ad hitlerum'],
    'Distraction': ["Misrepresentation of Someone's Position (Straw Man)", 'Presenting Irrelevant Data (Red Herring)', 'Whataboutism'],
    'Justification': ['Flag-waving', 'Appeal to authority', 'Bandwagon', 'Glittering generalities (Virtue)', 'Appeal to fear/prejudice'],
    'Simplification': ['Causal Oversimplification', 'Black-and-white Fallacy/Dictatorship'],
    'Manipulative Wording': ['Loaded Language', 'Obfuscation, Intentional vagueness, Confusion', 'Thought-terminating cliché', 'Exaggeration/Minimisation', 'Repetition'],
    'Call': ['Slogans']
}
T = networkx.DiGraph()

T.add_node("root")
T.add_edge("root", "propagandistic")
T.add_edge("root", "non-propagandistic")
T.add_edge("propagandistic", "Logos")
T.add_edge("Logos", "Repetition")
T.add_edge("Logos", "Obfuscation, Intentional vagueness, Confusion")
T.add_edge("Logos", "Reasoning")
T.add_edge("Logos", "Justification")
T.add_edge('Justification', "Slogans")
T.add_edge('Justification', "Bandwagon")
T.add_edge('Justification', "Appeal to authority")
T.add_edge('Justification', "Flag-waving")
T.add_edge('Justification', "Appeal to fear/prejudice")
T.add_edge('Reasoning', "Simplification")
T.add_edge('Simplification', "Causal Oversimplification")
T.add_edge('Simplification', "Black-and-white Fallacy/Dictatorship")
T.add_edge('Simplification', "Thought-terminating cliché")
T.add_edge('Reasoning', "Distraction")
T.add_edge('Distraction', "Misrepresentation of Someone's Position (Straw Man)")
T.add_edge('Distraction', "Presenting Irrelevant Data (Red Herring)")
T.add_edge('Distraction', "Whataboutism")
T.add_edge("propagandistic", "Ethos")
T.add_edge('Ethos', "Appeal to authority")
T.add_edge('Ethos', "Glittering generalities (Virtue)")
T.add_edge('Ethos', "Bandwagon")
T.add_edge('Ethos', "Ad Hominem")
T.add_edge('Ethos', "Transfer")
T.add_edge('Ad Hominem', "Doubt")
T.add_edge('Ad Hominem', "Name calling/Labeling")
T.add_edge('Ad Hominem', "Smears")
T.add_edge('Ad Hominem', "Reductio ad hitlerum")
T.add_edge('Ad Hominem', "Whataboutism")
T.add_edge("propagandistic", "Pathos")
T.add_edge('Pathos', "Exaggeration/Minimisation")
T.add_edge('Pathos', "Loaded Language")
T.add_edge('Pathos', "Appeal to (Strong) Emotions")
T.add_edge('Pathos', "Appeal to fear/prejudice")
T.add_edge('Pathos', "Flag-waving")

# Usage
# print_tree(hierarchical_tree, "root")
# print("hierarchical_tree: ", hierarchical_tree)

