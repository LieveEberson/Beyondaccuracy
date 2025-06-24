import json
import re

# For English, German, Swahili
def count_sentences(text, is_correct):
    sentences = re.split(r'[.!?]+(?:\s|$)', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return (len(sentences), is_correct)

# For Japanese
def count_japanese_sentences(text, is_correct):
    sentences = re.split(r'[\u3002\uff1f\uff01]+(?:\s|$)', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return (len(sentences), is_correct)

# For Russian
def count_russian_sentences(text, is_correct):
    sentences = re.split(r'[\u002E\u003F\u0021]+(?:\s|$)', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return (len(sentences), is_correct)

def count_thai_sentences(text, is_correct):
    sentence_end_patterns = (
        r'(\u0e04\u0e23\u0e31\u0e1a|'   # ครับ
        r'\u0e04\u0e48\u0e30|'          # ค่ะ
        r'\u0e04\u0e30|'                # คะ
        r'\u0e19\u0e30|'                # นะ
        r'\u0e2a\u0e34|'                # สิ
        r'\u0e08\u0e31\u0e07|'          # จัง
        r'\u0e40\u0e25\u0e22|'          # เลย
        r'\u0e21\u0e31\u0e49\u0e07|'    # มั้ง
        r'\u0e41\u0e25\u0e27|'    # แล้ว
        r'\u0e23\u0e37\u0e2d|'          # หรือ
        r'\u0e43\u0e0a\u0e48\u0e44\u0e2b\u0e21|'  # ใช่ไหม
        r'\u0e44\u0e07|'                # ไง
        r'\u0e01\u0e47\u0e44\u0e14\u0e49|'        # ก็ได้
        r'\u0e2b\u0e23\u0e2d\u0e01|'              # หรอก
        r'\u0e41\u0e2b\u0e25\u0e30|'              # แหละ
        r'\u0e40\u0e2d\u0e07)'                    # เอง
        r'(\s|$|[.!?])'
    )
    marked = re.sub(sentence_end_patterns, r'\1<SPLIT>', text)
    marked = re.sub(r'([.?!])(?=\s|[\u0e00-\u0e7f])', r'\1<SPLIT>', marked)
    sentences = [s.strip() for s in marked.split('<SPLIT>') if s.strip()]
    return (len(sentences), is_correct)

def load_language_file(lang):
    with open(f'{lang}_results.json', 'r') as f:
        return json.load(f)

def get_sentence_count(lang, text, is_correct):
    if lang in ['ja']:
        return count_japanese_sentences(text, is_correct)[0]
    elif lang in ['ru']:
        return count_russian_sentences(text, is_correct)[0]
    elif lang in ['th']:
        return count_thai_sentences(text, is_correct)[0]
    else:  # en, de, sw
        return count_sentences(text, is_correct)[0]

def calculate_language_pairs_incorrect_identical_sentence():
    languages = ['en', 'de', 'ru', 'ja', 'th', 'sw']
    lang_data = {lang: load_language_file(lang) for lang in languages}
    total_questions = len(lang_data['en'])
    pair_results = {}
    for i, lang1 in enumerate(languages):
        for lang2 in languages[i+1:]:
            both_incorrect_identical_sentence = 0
            for q in range(total_questions):
                is_correct1 = lang_data[lang1][q].get('is_correct', False)
                is_correct2 = lang_data[lang2][q].get('is_correct', False)
                pred1 = lang_data[lang1][q].get('predicted_answer', None)
                pred2 = lang_data[lang2][q].get('predicted_answer', None)
                count1 = get_sentence_count(lang1, lang_data[lang1][q].get('model_solution', ''), is_correct1)
                count2 = get_sentence_count(lang2, lang_data[lang2][q].get('model_solution', ''), is_correct2)
                if not is_correct1 and not is_correct2 and pred1 == pred2 and count1 == count2:
                    both_incorrect_identical_sentence += 1
            percentage = (both_incorrect_identical_sentence / total_questions) * 100
            pair_results[f"{lang1}-{lang2}"] = {
                'both_incorrect_identical_sentence': both_incorrect_identical_sentence,
                'percentage': percentage
            }
    print("\nLanguage Pair Analysis (Both Incorrect, Identical, Same Sentence Count):")
    print("Language Pair | Both Incorrect, Identical, Same Sentence | Percentage")
    print("-------------|------------------------------------------|-----------")
    for pair, data in pair_results.items():
        print(f"{pair:13} | {data['both_incorrect_identical_sentence']:40d} | {data['percentage']:9.2f}%")
    with open('language_pair_results_incorrect_identical_sentence.json', 'w') as f:
        json.dump(pair_results, f, indent=2)
    print("\nResults have been saved to 'language_pair_results_incorrect_identical_sentence.json'\n")

if __name__ == "__main__":
    calculate_language_pairs_incorrect_identical_sentence() 
