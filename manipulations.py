from enum import Enum
from nltk.corpus import wordnet as wn
import random

MAX_ITERATIONS = 50
MAX_LOOK_FOR_SPAN = 3


class Manipulations(Enum):
    # anything -> negation
    NEGATE_PART_PREMISE = 1
    # anything -> same thing
    SYNONYM = 2
    # anything -> negation
    ANTINOMY_PART_PREMISE = 3
    # anything -> same thing
    HYPONYM_PREMISE = 4
    # anything -> neutral
    SWITCH_DATA = 5
    # anything -> neutral
    SWITCH_PARTIAL_DATA = 6
    # anything -> entailment
    TAKE_PART_PREMISE = 7
    # entailment/negation -> opposite
    NEGATE_HYPOTHESIS = 8
    # anything -> same thing
    HYPERNYM_HYPOTHESIS = 9
    # anything -> negation
    IMPOSSIBILITY = 10
    # entailment -> entailment
    TRUNCATE_HYPOTHESIS = 11
    # anything -> entailment
    TAUTOLOGY = 12
    # anything -> entailment
    DUPLICATE_HYPOTHESIS = 13
    # entailment -> negation/entailment
    CHANGE_NUMBERS = 14


def negate_part_premise(sample):
    premise = sample['premise']
    span = extract_span(sample, 'premise')
    if span == -1:
        return None
    new_hypothesis = 'Is not true that ' + ' '.join(
        [word['rawText'] for word in sample['srl']['premise']['tokens'][:span]])
    return {'premise': premise, 'hypothesis': new_hypothesis, 'label': sample['label']}


def use_synonym(sample):
    part = random.choice(['premise', 'hypothesis'])
    candidates = []
    for word_info in sample['wsd'][part]:
        if word_info['nltkSynset'] is not None and word_info['nltkSynset'] != 'O':
            synset = wn.synset(word_info['nltkSynset'])
            if len(synset.lemmas()) > 1:
                candidates.append(word_info)
    if len(candidates) == 0:
        return None
    choice = random.choice(candidates)
    new_word = random.choice(wn.synset(choice['nltkSynset']).lemmas()).name()
    new_sample = {'premise': sample['premise'], 'hypothesis': sample['hypothesis'], 'label': sample['label']}
    new_sample[part] = ' '.join([
        new_word if word['index'] == choice['index'] else word['rawText'] for word in
        sample['srl'][part]['tokens']
    ])
    return new_sample


def use_antinomy(sample):
    span = sample['srl']['premise']['annotations'][0]['englishPropbank']['roles'][-1]['span'][-1]
    new_word = ''
    new_word_idx = -1
    for word in sample['wsd']['premise']:
        if word['pos'] == 'VERB' and word['nltkSynset'] != 'O' and word['nltkSynset'] is not None:
            i = 0
            while new_word == '' and i < len(wn.synsethypo(word['nltkSynset']).lemmas()):
                if wn.synset(word['nltkSynset']).lemmas()[i].antonyms():
                    new_word = random.choice(wn.synset(word['nltkSynset']).lemmas()[i].antonyms()).name()

                i += 1

        elif word['pos'] == 'AUX':
            new_word = word['text'] + ' not'

        if new_word != '':
            new_word_idx = word['index']
            break
    return {'premise': sample['premise'], 'label': sample['label'],
            'hypothesis': ' '.join([
                new_word if word['index'] == new_word_idx else word['rawText'] for word in
                sample['srl']['premise']['tokens'][:span]
            ])
            }


def use_hyponym(sample):
    candidates = []
    for word_info in sample['wsd']['premise']:
        if word_info['nltkSynset'] is not None and word_info['nltkSynset'] != 'O':
            synset = wn.synset(word_info['nltkSynset'])
            if len(synset.hyponyms()) > 0:
                candidates.append(word_info)
    if len(candidates) == 0:
        return None
    choice = random.choice(candidates)
    new_word = random.choice(random.choice(wn.synset(choice['nltkSynset']).hyponyms()).lemmas()).name()
    return {'hypothesis': sample['hypothesis'], 'label': sample['label'],
            'premise': ' '.join([
                new_word if word['index'] == choice['index'] else word['rawText'] for word in
                sample['srl']['premise']['tokens']
            ])
            }


def switch_data(sample1, data, samples):
    sample2 = samples[extract_sample(data)]
    return {'premise': sample1['premise'], 'hypothesis': sample2['hypothesis'], 'label': 'NEUTRAL'}


def extract_sample(data: dict):
    """
    Extracts a sample from a dictionary based on the keys.

    :param data: dictionary
    :param keys: list of keys
    :return: dictionary
    """
    extraction_done = False
    i = 0
    sample = None
    while not extraction_done:
        sample = random.randint(0, len(data) - 1)
        rand = random.random()
        if data[sample] > rand or i >= MAX_ITERATIONS:
            extraction_done = True
            data[sample] /= 2
        i += 1
    return sample

def extract_span(sample, part):
    span = -1
    i = 0
    while span == -1 and i < MAX_LOOK_FOR_SPAN:
        if (len(sample['srl'][part]['annotations'][0]['englishPropbank']['roles']) != 0):
            span = sample['srl'][part]['annotations'][0]['englishPropbank']['roles'][-1]['span'][-1]
        i += 0
    return span


def switch_partial_data(sample1, data, samples):
    sample2 = samples[extract_sample(data)]
    rnd = random.randint(0, 1)
    samples = [sample1, sample2]
    # premise --------
    sample = samples[rnd]
    new_sample_premise = random.choice(['premise', 'hypothesis'])
    if new_sample_premise == 'premise':
        span = extract_span(sample, 'premise')
        if span == -1:
            return None
        new_premise = ' '.join([word['rawText'] for word in sample['srl']['premise']['tokens'][:span]])
    else:
        new_premise = sample['hypothesis']
    # hypothesis --------
    sample = samples[1 - rnd]
    new_sample_hypothesis = random.choice(['premise', 'hypothesis'])
    if new_sample_hypothesis == 'premise':
        span = extract_span(sample, 'premise')
        if span == -1:
            return None
        new_hypothesis = ' '.join([word['rawText'] for word in sample['srl']['premise']['tokens'][:span]])
    else:
        new_hypothesis = sample['hypothesis']

    return {'premise': new_premise, 'hypothesis': new_hypothesis, 'label': 'NEUTRAL'}


def take_part_premise(sample):
    span = extract_span(sample, 'premise')
    new_hypothesis = ' '.join([word['rawText'] for word in sample['srl']['premise']['tokens'][:span]])
    return {'premise': sample['premise'], 'hypothesis': new_hypothesis, 'label': 'ENTAILMENT'}


def negate_hypothesis(sample):
    if sample['label'] == 'NEUTRAL':
        print("ERROR: NEGATE_HYPOTHESIS called on NEUTRAL sample")
        exit(1)
    new_hypothesis = 'Is not true that ' + sample['hypothesis']
    return {'premise': sample['premise'], 'hypothesis': new_hypothesis,
            'label': 'NEGATION' if sample['label'] == 'ENTAILMENT' else 'ENTAILMENT'}


def hypernym_hypothesis(sample):
    candidates = []
    for word_info in sample['wsd']['hypothesis']:
        if word_info['nltkSynset'] is not None and word_info['nltkSynset'] != 'O':
            synset = wn.synset(word_info['nltkSynset'])
            if len(synset.hypernyms()) > 0:
                candidates.append(word_info)
    if len(candidates) == 0:
        return None
    choice = random.choice(candidates)
    new_word = random.choice(random.choice(wn.synset(choice['nltkSynset']).hypernyms()).lemmas()).name()
    return {'premise': sample['premise'], 'label': sample['label'],
            'hypothesis': ' '.join([
                new_word if word['index'] == choice['index'] else word['rawText'] for word in
                sample['srl']['hypothesis']['tokens']
            ])
            }


def impossibility(sample):
    adj = get_random_adjective()
    noun = get_random_noun()
    new_hypothesis = f"A {adj} {noun} is not {adj}."
    return {'premise': sample['premise'], 'hypothesis': new_hypothesis, 'label': 'NEGATION'}


def get_random_adjective():
    adjectives = list(wn.all_synsets(wn.ADJ))
    adj = random.choice(adjectives)
    return adj.lemmas()[0].name()


def get_random_noun():
    nouns = list(wn.all_synsets(wn.NOUN))
    noun = random.choice(nouns)
    return noun.lemmas()[0].name()


def truncate_hypothesis(sample):
    span = extract_span(sample, 'hypothesis')
    if span == len(sample['srl']['hypothesis']['tokens']):
        return None
    new_hypothesis = ' '.join([word['rawText'] for word in sample['srl']['hypothesis']['tokens'][:span]])
    return {'premise': sample['premise'], 'hypothesis': new_hypothesis, 'label': 'ENTAILMENT'}


def tautology(sample):
    adj = get_random_adjective()
    noun = get_random_noun()
    new_hypothesis = f"A {adj} {noun} is {adj}."
    return {'premise': sample['premise'], 'hypothesis': new_hypothesis, 'label': 'ENTAILMENT'}


def duplicate_hypothesis(sample):
    return {'premise': sample['hypothesis'], 'hypothesis': sample['hypothesis'], 'label': 'ENTAILMENT'}


def change_numbers(sample, numeric_id, comparator, chosen_manipulation):
    old_num = int(sample['wsd']['hypothesis'][numeric_id]['text'])
    if comparator == 1 and chosen_manipulation == 'ENTAILMENT':
        new_num = random.randint(0, old_num - 1)
    elif comparator == -1 and chosen_manipulation == 'ENTAILMENT':
        new_num = random.randint(old_num + 1, old_num + 1000)
    elif comparator == 0:
        new_num = random.choice([random.randint(0, old_num - 1), random.randint(old_num + 1, old_num + 1000)])
    elif comparator == 1:
        new_num = random.randint(old_num * 100 + 1, old_num * 100 + 1000)
    else:
        new_num = random.randint(0, old_num // 100)

    return {'premise': sample['premise'], 'hypothesis': ' '.join([
        str(new_num) if word['index'] == numeric_id else word['rawText'] for word in
        sample['srl']['hypothesis']['tokens']
    ]), 'label': 'NEGATION'}


def exec_manipulation(sample, manipulation, manipulation_output, numeric, data, samples):
    if manipulation == Manipulations.NEGATE_PART_PREMISE:
        return negate_part_premise(sample)
    elif manipulation == Manipulations.SYNONYM:
        return use_synonym(sample)
    elif manipulation == Manipulations.ANTINOMY_PART_PREMISE:
        return use_antinomy(sample)
    elif manipulation == Manipulations.HYPONYM_PREMISE:
        return use_hyponym(sample)
    elif manipulation == Manipulations.SWITCH_DATA:
        return switch_data(sample, data, samples)
    elif manipulation == Manipulations.SWITCH_PARTIAL_DATA:
        return switch_partial_data(sample, data, samples)
    elif manipulation == Manipulations.TAKE_PART_PREMISE:
        return take_part_premise(sample)
    elif manipulation == Manipulations.NEGATE_HYPOTHESIS:
        return negate_hypothesis(sample)
    elif manipulation == Manipulations.HYPERNYM_HYPOTHESIS:
        return hypernym_hypothesis(sample)
    elif manipulation == Manipulations.IMPOSSIBILITY:
        return impossibility(sample)
    elif manipulation == Manipulations.TRUNCATE_HYPOTHESIS:
        return truncate_hypothesis(sample)
    elif manipulation == Manipulations.TAUTOLOGY:
        return tautology(sample)
    elif manipulation == Manipulations.DUPLICATE_HYPOTHESIS:
        return duplicate_hypothesis(sample)
    elif manipulation == Manipulations.CHANGE_NUMBERS:
        return change_numbers(sample, *numeric, manipulation_output)
    else:
        print("ERROR: unknown manipulation")
        exit(1)
