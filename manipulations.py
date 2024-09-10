from enum import Enum
import nltk
from nltk.corpus import wordnet as wn
import random

MAX_ITERATIONS = 50


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
    # anything -> entailment / negation
    TALK_ABOUT = 13
    # anything -> entailment
    DUPLICATE_HYPOTHESIS = 14
    #anything -> neg
    NOT_EXIST = 15


def negate_part_premise(sample):
    premise = sample['premise']
    span = sample['srl']['premise']['annotations'][0]['englishPropbank']['roles'][-1]['span'][-1]
    new_hypothesis = 'Is not true that ' + ' '.join(
        [word['rawText'] for word in sample['srl']['premise']['tokens'][:span]])
    return {'premise': premise, 'hypothesis': new_hypothesis, 'label': sample['label']}


def use_synonym(sample):
    part = random.choice(['premise', 'hypothesis'])
    candidates = []
    for word_info in sample['wsd'][part]:
        if word_info['nltkSynset'] is not None and word_info['word'] != 'O':
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
        if word_info['nltkSynset'] is not None and word_info['word'] != 'O':
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


def switch_data(sample1, data, keys):
    sample2 = extract_sample(data, keys)
    return {'premise': sample1['premise'], 'hypothesis': sample2['hypothesis'], 'label': 'NEUTRAL'}


def extract_sample(data: dict, keys: list):
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
        sample = random.choice(keys)
        rand = random.random()
        if data[sample] > rand or i >= MAX_ITERATIONS:
            extraction_done = True
            data[sample] /= 2
        i += 1
    return sample


def switch_partial_data(sample1, data, keys):
    sample2 = extract_sample(data, keys)
    rnd = random.randint(0, 1)
    samples = [sample1, sample2]
    # premise --------
    sample = samples[rnd]
    new_sample_premise = random.choice(['premise', 'hypothesis'])
    if new_sample_premise == 'premise':
        span = sample['srl']['premise']['annotations'][0]['englishPropbank']['roles'][-1]['span'][-1]
        new_premise = ' '.join([word['rawText'] for word in sample['srl']['premise']['tokens'][:span]])
    else:
        new_premise = sample['hypothesis']
    # hypothesis --------
    sample = samples[1 - rnd]
    new_sample_hypothesis = random.choice(['premise', 'hypothesis'])
    if new_sample_hypothesis == 'premise':
        span = sample['srl']['premise']['annotations'][0]['englishPropbank']['roles'][-1]['span'][-1]
        new_hypothesis = ' '.join([word['rawText'] for word in sample['srl']['premise']['tokens'][:span]])
    else:
        new_hypothesis = sample['hypothesis']

    return {'premise': new_premise, 'hypothesis': new_hypothesis, 'label': 'NEUTRAL'}


def take_part_premise(sample):
    span = sample['srl']['premise']['annotations'][0]['englishPropbank']['roles'][-1]['span'][-1]
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
        if word_info['nltkSynset'] is not None and word_info['word'] != 'O':
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
    span = sample['srl']['hypothesis']['annotations'][0]['englishPropbank']['roles'][-1]['span'][-1]
    new_hypothesis = ' '.join([word['rawText'] for word in sample['srl']['hypothesis']['tokens'][:span]])
    return {'premise': sample['premise'], 'hypothesis': new_hypothesis, 'label': 'ENTAILMENT'}



