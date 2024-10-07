from manipulations import *
import sys
import nltk
import concurrent.futures
from datasets import Dataset

PRODUCE_ENTAILMENT_LIST = [Manipulations.TAKE_PART_PREMISE, Manipulations.TRUNCATE_HYPOTHESIS, Manipulations.TAUTOLOGY,
                           Manipulations.NEGATE_HYPOTHESIS, Manipulations.DUPLICATE_HYPOTHESIS]
PRODUCE_NEUTRAL_LIST = [Manipulations.SWITCH_DATA, Manipulations.SWITCH_PARTIAL_DATA, Manipulations.CHANGE_NUMBERS]
PRODUCE_CONTRADICTION_LIST = [Manipulations.NEGATE_PART_PREMISE, Manipulations.ANTINOMY_PART_PREMISE,
                         Manipulations.IMPOSSIBILITY, Manipulations.NEGATE_HYPOTHESIS, Manipulations.CHANGE_NUMBERS]
PRODUCE_ANYTHING_LIST = [Manipulations.SYNONYM, Manipulations.HYPONYM_PREMISE, Manipulations.HYPERNYM_HYPOTHESIS]

MAX_TRIALS_SAMPLE = 3

MAJORITY_COMPARATORS = ['more', 'larger', 'higher', 'longer', 'taller', 'older', 'faster']
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
        manipulations_list += PRODUCE_CONTRADICTION_LIST
        proportions[1] += 1
        manipulation_output = 'CONTRADICTION'
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
    premises = []
    hypotheses = []
    labels = []
    new_premises = []
    new_hypotheses = []
    new_labels = []
    proportions = [0, 0, 0]
    indices = {}
    augment_methods = []
    old_premises = []
    old_hypotheses = []
    manipulation_info = {}
    old_labels = []
    for i, sample in enumerate(data):
        print_progress_bar(i / len(data), text=f" Processing initial dataset ")
        premises.append(sample['premise'])
        hypotheses.append(sample['hypothesis'])
        labels.append(sample['label'])
        indices[i] = 1
        if sample['label'] == 'ENTAILMENT':
            proportions[0] += 1
        elif sample['label'] == 'CONTRADICTION':
            proportions[1] += 1
        else:
            proportions[2] += 1
    print("\n", proportions)
    for i in range(num_new_samples):
        print_progress_bar(i / num_new_samples, text=f" Augmenting data ")
        old_sample = data[extract_sample(indices)]
        sample = old_sample.copy()
        produced = False
        trials = 0
        while not produced:
            manipulation, output, numeric_info = choose_manipulation(sample, proportions)
            if manipulation.name not in manipulation_info:
                manipulation_info[manipulation.name] = {'count': 0, 'success': 0}
            manipulation_info[manipulation.name]['count'] += 1
            new_sample = exec_manipulation(sample, manipulation, output, numeric_info, indices, data)
            produced = new_sample is not None or trials > MAX_TRIALS_SAMPLE
            trials += 1
        if new_sample is not None:
            old_premises.append(old_sample['premise'])
            old_hypotheses.append(old_sample['hypothesis'])
            old_labels.append(old_sample['label'])
            manipulation_info[manipulation.name]['success'] += 1
            new_premises.append(new_sample['premise'])
            new_hypotheses.append(new_sample['hypothesis'])
            new_labels.append(new_sample['label'])
            augment_methods.append(manipulation.name)

    new_data_dataset = Dataset.from_dict({
        'premise': new_premises,
        'hypothesis': new_hypotheses,
        'label': new_labels,
        'augment_method': augment_methods,
        'old_premise': old_premises,
        'old_hypothesis': old_hypotheses,
        'old_label': old_labels
    })

    data = Dataset.from_dict({
        'premise': premises + new_premises,
        'hypothesis': hypotheses + new_hypotheses,
        'label': labels + new_labels
    })

    return data, new_data_dataset, manipulation_info


def augment_data_multithread(data : Dataset, num_new_samples: int):
    nltk.download('wordnet')
    premises = []
    hypotheses = []
    labels = []
    new_premises = []
    new_hypotheses = []
    new_labels = []
    proportions = [0, 0, 0]
    indices = {}

    # Primo for, non c'è bisogno di parallelizzarlo
    for i, sample in enumerate(data):
        premises.append(sample['premise'])
        hypotheses.append(sample['hypothesis'])
        labels.append(sample['label'])
        indices[i] = 1
        if sample['label'] == 'ENTAILMENT':
            proportions[0] += 1
        elif sample['label'] == 'CONTRADICTION':
            proportions[1] += 1
        else:
            proportions[2] += 1

    # Funzione da eseguire in parallelo
    def augment_sample(i):
        sample = data[extract_sample(indices)]  # Estrae un campione dalla lista originale
        manipulation, output, numeric_info = choose_manipulation(sample, proportions)
        new_sample = exec_manipulation(sample, manipulation, output, numeric_info, indices, data)

        # Ritorna i valori delle colonne del nuovo campione
        return new_sample['premise'], new_sample['hypothesis'], new_sample['label']

    # Parallelizzazione del secondo for
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(augment_sample, range(num_new_samples))

    # Popola le liste new_premises, new_hypotheses, new_labels con i risultati
    for result in results:
        if result is not None:
            premise, hypothesis, label = result
            new_premises.append(premise)
            new_hypotheses.append(hypothesis)
            new_labels.append(label)

    # Crea il dataset per i nuovi campioni
    new_data_dataset = Dataset.from_dict({
        'premise': new_premises,
        'hypothesis': new_hypotheses,
        'label': new_labels
    })

    # Crea il dataset completo concatenando i vecchi e nuovi campioni
    data = Dataset.from_dict({
        'premise': premises + new_premises,
        'hypothesis': hypotheses + new_hypotheses,
        'label': labels + new_labels
    })

    return data, new_data_dataset



def print_progress_bar(percentuale: float, lunghezza_barra: int = 30, text: str="") -> None:
    blocchi_compilati = int(lunghezza_barra * percentuale)
    barra = "[" + "=" * (blocchi_compilati - 1) + ">" + " " * (lunghezza_barra - blocchi_compilati) + "]"
    sys.stdout.write(f"\r{barra} {percentuale * 100:.2f}% complete " + text)
    sys.stdout.flush()