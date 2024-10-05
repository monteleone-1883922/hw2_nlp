from manipulations import *
import sys
import nltk
import concurrent.futures
from datasets import Dataset

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
        manipulation_output = 'ENTAILMENT'
        proportions[0] += 1
        numeric_id, comparator = isNumeric(sample)

    elif rnd < (proportions[0] + proportions[1]) / sum_proportions:
        manipulations_list += PRODUCE_NEGATION_LIST
        proportions[1] += 1
        manipulation_output = 'NEGATION'
        numeric_id, comparator = isNumeric(sample)
    else:
        manipulations_list += PRODUCE_NEUTRAL_LIST
        proportions[2] += 1
        manipulation_output = 'NEUTRAL'

    for manipulation in manipulations_list:
        if not (label != 'ENTAILMENT' and (
                manipulation == Manipulations.TRUNCATE_HYPOTHESIS or manipulation == Manipulations.CHANGE_NUMBERS)) and \
                not ((label == manipulation_output or label == 'NEUTRAL') and manipulation == Manipulations.NEGATE_HYPOTHESIS) and \
                not (manipulation == Manipulations.CHANGE_NUMBERS and numeric_id == -1) and \
                not (manipulation_output == 'ENTAILMENT' and manipulation == Manipulations.CHANGE_NUMBERS and comparator == 0):
            manipulation_values.append(manipulation)

    if label == manipulation_output:
        manipulation_values += PRODUCE_ANYTHING_LIST

    return random.choice(manipulation_values), manipulation_output, (numeric_id, comparator)


def isNumeric(sample):
    numeric_id = -1
    comparator = 0
    for word_info in sample['wsd']['hypothesis']:
        if word_info['pos'] == 'NUM':
            numeric_id = word_info['index']
            break
        elif word_info['text'] in MAJORITY_COMPARATORS:
            comparator = 1
        elif word_info['text'] in MINORITY_COMPARATORS:
            comparator = -1
    return numeric_id, comparator

def augment_data(data: Dataset, num_new_samples: int):
    nltk.download('wordnet')
    new_data = []
    proportions = [0, 0, 0]
    indices = {}
    for i, sample in enumerate(data):
        indices[i] = 1
        if sample['label'] == 'ENTAILMENT':
            proportions[0] += 1
        elif sample['label'] == 'NEGATION':
            proportions[1] += 1
        else:
            proportions[2] += 1
    for i in range(num_new_samples):
        print_progress_bar(i / num_new_samples, text=f" Augmenting data ")
        sample = data[extract_sample(indices)]
        manipulation, output, numeric_info = choose_manipulation(sample, proportions)
        new_sample = exec_manipulation(sample, manipulation, output, numeric_info, indices, data)
        new_data.append(new_sample)

    new_data_dataset = Dataset.from_dict({
        'data': new_data  # Supponendo che results contenga nuovi campioni strutturati
    })

    # Concatena i dataset
    data = Dataset.concatenate([data, new_data_dataset])
    return data, new_data_dataset


def augment_data_multithread(data : Dataset, num_new_samples: int):
    nltk.download('wordnet')
    new_data = []
    proportions = [0, 0, 0]
    indices = {}

    # Primo for, non c'è bisogno di parallelizzarlo
    for i, sample in enumerate(data):
        indices[i] = 1
        if sample['label'] == 'ENTAILMENT':
            proportions[0] += 1
        elif sample['label'] == 'NEGATION':
            proportions[1] += 1
        else:
            proportions[2] += 1

    # Funzione da eseguire in parallelo
    def augment_sample(i):
        sample = data[extract_sample(indices)]
        manipulation, output, numeric_info = choose_manipulation(sample, proportions)
        new_sample = exec_manipulation(sample, manipulation, output, numeric_info, indices, data)
        return new_sample

    # Parallelizzazione del secondo for
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(augment_sample, range(num_new_samples))

    # Aggiungi i nuovi campioni ai dati originali

    new_data.extend(results)
    new_data_dataset = Dataset.from_dict({
        'data': new_data  # Supponendo che results contenga nuovi campioni strutturati
    })

    # Concatena i dataset
    data = Dataset.concatenate([data, new_data_dataset])
    return data, new_data_dataset



def print_progress_bar(percentuale: float, lunghezza_barra: int = 30, text: str="") -> None:
    blocchi_compilati = int(lunghezza_barra * percentuale)
    barra = "[" + "=" * (blocchi_compilati - 1) + ">" + " " * (lunghezza_barra - blocchi_compilati) + "]"
    sys.stdout.write(f"\r{barra} {percentuale * 100:.2f}% complete " + text)
    sys.stdout.flush()