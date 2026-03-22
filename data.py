import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from datasets import load_dataset
from collections import defaultdict

def load_clinc150():
    dataset = load_dataset('clinc_oos', 'plus')
    splits = {}
    for s in ['train', 'validation', 'test']:
        texts   = [x['text']   for x in dataset[s]]
        intents = [x['intent'] for x in dataset[s]]
        splits[s] = (texts, intents)

    label_feature = dataset['train'].features['intent']
    idx_to_intent = {i: label_feature.int2str(i) for i in range(label_feature.num_classes)}
    intent_to_idx = {v: k for k, v in idx_to_intent.items()}
    return splits, idx_to_intent, intent_to_idx

def load_glove(path='glove.6B.100d.txt'):
    glove = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            glove[parts[0]] = np.array(parts[1:], dtype=np.float32)
    return glove, 100

def text_to_embedding(text, glove, D):
    tokens = word_tokenize(text.lower())
    vecs = [glove[t] for t in tokens if t in glove]
    if not vecs:
        return np.zeros(D, dtype=np.float32)
    v = np.mean(vecs, axis=0)
    norm = np.linalg.norm(v)
    return (v / norm).astype(np.float32) if norm > 1e-12 else v

def build_embeddings(texts, glove, D):
    return np.array([text_to_embedding(t, glove, D) for t in texts], dtype=np.float32)

def build_domain_structure(intent_to_idx):
    CLINC150_DOMAINS = {
        'banking':           ['transfer','transactions','balance','freeze_account',
                              'pay_bill','bill_balance','bill_due','interest_rate',
                              'routing','min_payment','new_card','lost_card',
                              'card_decline','pin_change','report_fraud'],
        'credit_cards':      ['credit_score','report_lost_card','credit_limit',
                              'rewards_balance','application_status',
                              'card_about_to_expire','replacement_card_duration',
                              'expiration_date','credit_limit_change','damaged_card',
                              'improve_credit_score','apr','redeem_rewards',
                              'account_blocked','spending_history'],
        'kitchen_and_dining':['recipe','food_last','meal_suggestion','nutrition_info',
                              'calories','ingredient_substitution','cook_time',
                              'food_beverage_price','restaurant_reviews',
                              'restaurant_reservation','confirm_reservation',
                              'how_busy','cancel_reservation','accept_reservations',
                              'ingredients_list'],
        'home':              ['smart_home','shopping_list','shopping_list_update',
                              'next_song','play_music','update_playlist','todo_list',
                              'todo_list_update','calendar','calendar_update',
                              'order','order_status','reminder','reminder_update',
                              'what_can_i_ask'],
        'auto_and_commute':  ['car_rental','car_bluetooth','tire_pressure',
                              'oil_change_when','oil_change_how','jump_start',
                              'uber','schedule_maintenance','last_maintenance',
                              'insurance','traffic','directions','gas','gas_type',
                              'distance'],
        'travel':            ['book_flight','book_hotel','get_hotel_recommendations',
                              'travel_suggestion','travel_notification',
                              'carry_on_baggage','timezone','international_visa',
                              'plug_type','exchange_rate','flight_status',
                              'international_fees','vaccines','lost_luggage','mpg'],
        'utility':           ['time','alarm','timer','weather','date','find_phone',
                              'share_location','current_location',
                              'meeting_schedule','calculator',
                              'measurement_conversion','spelling','definition',
                              'change_accent','sync_device'],
        'work':              ['direct_deposit','pto_request','taxes','payday','w2',
                              'income','rollover_401k','find_internship','fico_score',
                              'insurance_change','user_name','password_reset',
                              'change_user_name','change_password','next_holiday'],
        'meta':              ['who_do_you_work_for','do_you_have_pets','are_you_a_bot',
                              'meaning_of_life','who_made_you','thank_you','goodbye',
                              'tell_joke','where_are_you_from','how_old_are_you',
                              'what_is_your_name','what_are_your_hobbies','fun_fact',
                              'change_ai_name','what_can_i_ask'],
        'small_talk':        ['greeting','yes','no','maybe','I_am_bored',
                              'flip_coin','roll_dice','laugh','story','text',
                              'repeat','whisper_mode','make_call','number_facts',
                              'next_holiday'],
    }
    intent_to_domain = {}
    for domain, names in CLINC150_DOMAINS.items():
        for name in names:
            if name in intent_to_idx:
                intent_to_domain[name] = domain
    domain_to_intents = defaultdict(list)
    for intent, domain in intent_to_domain.items():
        domain_to_intents[domain].append(intent_to_idx[intent])
    return domain_to_intents

def build_balanced_quads(domain_to_intents, train_labels, max_per_family=30, seed=42):
    rng = np.random.default_rng(seed)
    all_quads = []
    for domain, idxs in domain_to_intents.items():
        if len(idxs) < 2: continue
        pairs = []
        for ia in range(len(idxs)):
            for ib in range(ia+1, len(idxs)):
                ex_a = np.where(train_labels == idxs[ia])[0]
                ex_b = np.where(train_labels == idxs[ib])[0]
                if len(ex_a) == 0 or len(ex_b) == 0: continue
                # Pick a few examples for each pair of intents in the domain
                for i in range(min(3, len(ex_a), len(ex_b))):
                    pairs.append((int(ex_a[i]), int(ex_b[i])))

        # Form quads from pairs
        if len(pairs) < 2: continue
        family_quads = []
        for i in range(len(pairs)):
            for j in range(i+1, len(pairs)):
                ai, bi = pairs[i]
                ci, di = pairs[j]
                family_quads.append((ai, bi, ci, di))
                family_quads.append((ci, di, ai, bi))

        if len(family_quads) > max_per_family:
            sel = rng.choice(len(family_quads), max_per_family, replace=False)
            family_quads = [family_quads[k] for k in sel]
        all_quads.extend(family_quads)
    return all_quads
