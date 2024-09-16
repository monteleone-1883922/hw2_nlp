import random
from manipulations import Manipulations

PRODUCE_ENTAILMENT_LIST = [Manipulations.TAKE_PART_PREMISE, Manipulations.TRUNCATE_HYPOTHESIS, Manipulations.TAUTOLOGY,
                           Manipulations.NEGATE_HYPOTHESIS, Manipulations.DUPLICATE_HYPOTHESIS]
PRODUCE_NEUTRAL_LIST = [Manipulations.SWITCH_DATA, Manipulations.SWITCH_PARTIAL_DATA]
PRODUCE_NEGATION_LIST = [Manipulations.NEGATE_PART_PREMISE, Manipulations.ANTINOMY_PART_PREMISE,
                         Manipulations.IMPOSSIBILITY, Manipulations.NEGATE_HYPOTHESIS, Manipulations.CHANGE_NUMBERS]
PRODUCE_ANYTHING_LIST = [Manipulations.SYNONYM, Manipulations.HYPONYM_PREMISE, Manipulations.HYPERNYM_HYPOTHESIS]


def choose_manipulation(sample, proportions: list):
    manipulation_values = []
    label = sample['label']
    rnd = random.random()
    manipulations_list = []
    sum_proportions = sum(proportions)
    if rnd < proportions[0] / sum_proportions:
        manipulations_list += PRODUCE_ENTAILMENT_LIST
        chosen_manipulation = 'ENTAILMENT'
        proportions[0] += 1
    elif rnd < (proportions[0] + proportions[1]) / sum_proportions:
        manipulations_list += PRODUCE_NEGATION_LIST
        proportions[1] += 1
        chosen_manipulation = 'NEGATION'
    else:
        manipulations_list += PRODUCE_NEUTRAL_LIST
        proportions[2] += 1
        chosen_manipulation = 'NEUTRAL'

    # not ((label == 'ENTAILMENT' or label == 'NEGATION') and manipulation == Manipulations.NEGATE_HYPOTHESIS) and \
    for manipulation in manipulations_list:
        if not (label != 'ENTAILMENT' and manipulation == Manipulations.TRUNCATE_HYPOTHESIS) and \
                not ((label == chosen_manipulation or label == 'NEUTRAL') and manipulation == Manipulations.NEGATE_HYPOTHESIS):
            manipulation_values.append(manipulation)

    if label == chosen_manipulation:
        manipulation_values += PRODUCE_ANYTHING_LIST

    return random.choice(manipulation_values)
