from collections import deque
from networkx import DiGraph
import networkx
from sklearn_hierarchical_classification.constants import ROOT
import sys
sys.path.append('.')

# Create the tree structure by duplicating nodes with multiple parents
T = networkx.DiGraph()

T.add_node("root")
T.add_edge("root", "propagandistic")
T.add_edge("root", "non-propagandistic")

# Main categories under propagandistic
T.add_edge("propagandistic", "Logos")
T.add_edge("propagandistic", "Ethos") 
T.add_edge("propagandistic", "Pathos")

# Logos branch
T.add_edge("Logos", "Repetition")
T.add_edge("Logos", "Obfuscation, Intentional vagueness, Confusion")
T.add_edge("Logos", "Reasoning")
T.add_edge("Logos", "Justification")

# Justification under Logos
T.add_edge('Justification', "Logos_Slogans")  # Renamed to avoid conflict
T.add_edge('Justification', "Logos_Bandwagon")  # Duplicate for Logos
T.add_edge('Justification', "Logos_Appeal_to_authority")  # Duplicate for Logos
T.add_edge('Justification', "Logos_Flag-waving")  # Duplicate for Logos
T.add_edge('Justification', "Logos_Appeal_to_fear/prejudice")  # Duplicate for Logos

# Reasoning branch
T.add_edge('Reasoning', "Simplification")
T.add_edge('Simplification', "Causal Oversimplification")
T.add_edge('Simplification', "Black-and-white Fallacy/Dictatorship")
T.add_edge('Simplification', "Thought-terminating cliché")

T.add_edge('Reasoning', "Distraction")
T.add_edge('Distraction', "Misrepresentation of Someone's Position (Straw Man)")
T.add_edge('Distraction', "Presenting Irrelevant Data (Red Herring)")
T.add_edge('Distraction', "Distraction_Whataboutism")  # Separate node for Distraction

# Ethos branch
T.add_edge('Ethos', "Ethos_Appeal_to_authority")  # Duplicate for Ethos
T.add_edge('Ethos', "Glittering generalities (Virtue)")
T.add_edge('Ethos', "Ethos_Bandwagon")  # Duplicate for Ethos
T.add_edge('Ethos', "Ad Hominem")
T.add_edge('Ethos', "Transfer")

# Ad Hominem branch
T.add_edge('Ad Hominem', "Doubt")
T.add_edge('Ad Hominem', "Name calling/Labeling")
T.add_edge('Ad Hominem', "Smears")
T.add_edge('Ad Hominem', "Reductio ad hitlerum")
T.add_edge('Ad Hominem', "AdHominem_Whataboutism")  # Separate node for Ad Hominem

# Pathos branch
T.add_edge('Pathos', "Exaggeration/Minimisation")
T.add_edge('Pathos', "Loaded Language")
T.add_edge('Pathos', "Appeal to (Strong) Emotions")
T.add_edge('Pathos', "Pathos_Appeal_to_fear/prejudice")  # Duplicate for Pathos
T.add_edge('Pathos', "Pathos_Flag-waving")  # Duplicate for Pathos

# Create mapping for original labels to duplicated nodes
LABEL_MAPPING = {
    "Whataboutism": ["Distraction_Whataboutism", "AdHominem_Whataboutism"],
    "Appeal to authority": ["Logos_Appeal_to_authority", "Ethos_Appeal_to_authority"],
    "Bandwagon": ["Logos_Bandwagon", "Ethos_Bandwagon"],
    "Flag-waving": ["Logos_Flag-waving", "Pathos_Flag-waving"],
    "Appeal to fear/prejudice": ["Logos_Appeal_to_fear/prejudice", "Pathos_Appeal_to_fear/prejudice"],
    "Slogans": ["Logos_Slogans"]  # Only appears under Logos in this structure
}