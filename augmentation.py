import random
from manipulations import Manipulations

PRODUCE_ENTAILMENT_LIST = [Manipulations.TAKE_PART_PREMISE, Manipulations.TRUNCATE_HYPOTHESIS, Manipulations.TAUTOLOGY,
                           Manipulations.NEGATE_HYPOTHESIS, Manipulations.DUPLICATE_HYPOTHESIS]
PRODUCE_NEUTRAL_LIST = [Manipulations.SWITCH_DATA, Manipulations.SWITCH_PARTIAL_DATA, Manipulations.CHANGE_NUMBERS]
PRODUCE_NEGATION_LIST = [Manipulations.NEGATE_PART_PREMISE, Manipulations.ANTINOMY_PART_PREMISE,
                         Manipulations.IMPOSSIBILITY, Manipulations.NEGATE_HYPOTHESIS, Manipulations.CHANGE_NUMBERS]
PRODUCE_ANYTHING_LIST = [Manipulations.SYNONYM, Manipulations.HYPONYM_PREMISE, Manipulations.HYPERNYM_HYPOTHESIS]


MAJORITY_COMPARATORS = ['more', 'larger', 'higher', 'longer', 'taller', 'older']
MINORITY_COMPARATORS = ['less', 'fewer', 'smaller', 'lower', 'shorter', 'younger']


def choose_manipulation(sample, proportions: list):
    manipulation_values = []
    label = sample['label']
    rnd = random.random()
    manipulations_list = []
    numeric_id = -1
    comparator = 0
    sum_proportions = sum(proportions)
    if rnd < proportions[0] / sum_proportions:
        manipulations_list += PRODUCE_ENTAILMENT_LIST
        chosen_manipulation = 'ENTAILMENT'
        proportions[0] += 1
        for word_info in sample['wsd']['hypothesis']:
            if word_info['pos'] == 'NUM':
                numeric_id = word_info['index']
                break
            elif word_info['rawText'] in MAJORITY_COMPARATORS:
                comparator = 1
            elif word_info['rawText'] in MINORITY_COMPARATORS:
                comparator = -1
    elif rnd < (proportions[0] + proportions[1]) / sum_proportions:
        manipulations_list += PRODUCE_NEGATION_LIST
        proportions[1] += 1
        chosen_manipulation = 'NEGATION'
        for word_info in sample['wsd']['hypothesis']:
            if word_info['pos'] == 'NUM':
                numeric_id = word_info['index']
                break
            elif word_info['rawText'] in MAJORITY_COMPARATORS:
                comparator = 1
            elif word_info['rawText'] in MINORITY_COMPARATORS:
                comparator = -1
    else:
        manipulations_list += PRODUCE_NEUTRAL_LIST
        proportions[2] += 1
        chosen_manipulation = 'NEUTRAL'




    for manipulation in manipulations_list:
        if not (label != 'ENTAILMENT' and (manipulation == Manipulations.TRUNCATE_HYPOTHESIS or manipulation == Manipulations.CHANGE_NUMBERS)) and \
                not ((label == chosen_manipulation or label == 'NEUTRAL') and manipulation == Manipulations.NEGATE_HYPOTHESIS) and \
                not (manipulation == Manipulations.CHANGE_NUMBERS and numeric_id == -1) and \
                not (chosen_manipulation == 'ENTAILMENT' and manipulation == Manipulations.CHANGE_NUMBERS and comparator == 0):
            manipulation_values.append(manipulation)

    if label == chosen_manipulation:
        manipulation_values += PRODUCE_ANYTHING_LIST

    return random.choice(manipulation_values), chosen_manipulation, (numeric_id, comparator)
