import json
import re

# For English, German, Swahili
def count_sentences(text, is_correct):
    # Split text into sentences using common sentence endings
    # This pattern looks for periods, question marks, or exclamation marks followed by space or end of string
    sentences = re.split(r'[.!?]+(?:\s|$)', text)
    # Filter out empty strings that might result from the split
    sentences = [s.strip() for s in sentences if s.strip()]
    return (len(sentences), is_correct)

# For Japanese
def count_japanese_sentences(text, is_correct):
    # Split text into sentences using Japanese punctuation marks
    sentences = re.split(r'[\u3002\uff1f\uff01]+(?:\s|$)', text)
    # Filter out empty strings that might result from the split
    sentences = [s.strip() for s in sentences if s.strip()]
    return (len(sentences), is_correct)

# For Russian
def count_russian_sentences(text, is_correct):
    # Split text into sentences using Russian punctuation marks
    sentences = re.split(r'[\u002E\u003F\u0021]+(?:\s|$)', text)
    # Filter out empty strings that might result from the split
    sentences = [s.strip() for s in sentences if s.strip()]
    return (len(sentences), is_correct)

def count_thai_sentences(text, is_correct):
    # Sentence-ending particles (in Unicode)
    sentence_end_patterns = (
        r'(\u0e04\u0e23\u0e31\u0e1a|'   # ครับ
        r'\u0e04\u0e48\u0e30|'          # ค่ะ
        r'\u0e04\u0e30|'                # คะ
        r'\u0e19\u0e30|'                # นะ
        r'\u0e2a\u0e34|'                # สิ
        r'\u0e08\u0e31\u0e07|'          # จัง
        r'\u0e40\u0e25\u0e22|'          # เลย
        r'\u0e21\u0e31\u0e49\u0e07|'    # มั้ง
        r'\u0e41\u0e25\u0e49\u0e27|'    # แล้ว
        r'\u0e23\u0e37\u0e2d|'          # หรือ
        r'\u0e43\u0e0a\u0e48\u0e44\u0e2b\u0e21|'  # ใช่ไหม
        r'\u0e44\u0e07|'                # ไง
        r'\u0e01\u0e47\u0e44\u0e14\u0e49|'        # ก็ได้
        r'\u0e2b\u0e23\u0e2d\u0e01|'              # หรอก
        r'\u0e41\u0e2b\u0e25\u0e30|'              # แหละ
        r'\u0e40\u0e2d\u0e07)'                    # เอง
        r'(\s|$|[.!?])'
    )

    # Add <SPLIT> marker after sentence enders
    marked = re.sub(sentence_end_patterns, r'\1<SPLIT>', text)

    # Also split on Western punctuation, if followed by Thai char or space
    marked = re.sub(r'([.?!])(?=\s|[\u0e00-\u0e7f])', r'\1<SPLIT>', marked)

    # Split and clean
    sentences = [s.strip() for s in marked.split('<SPLIT>') if s.strip()]
    return (len(sentences), is_correct)

def combine_language_results(results):
    """
    Combine results from all languages for each index number.
    Returns a list of dictionaries containing the sentence counts and correctness
    for each language at each index.
    """
    combined_results = []
    
    # Get the number of items (assuming all languages have the same number of items)
    num_items = len(results[0][1])
    
    # For each index
    for i in range(num_items):
        item_result = {
            'question_number': i + 1  # Add question number (1-based indexing)
        }
        # For each language
        for lang, lang_results in results:
            count, correct = lang_results[i]
            item_result[lang] = {
                'sentence_count': count,
                'is_correct': correct
            }
        combined_results.append(item_result)
    
    return combined_results

if __name__ == "__main__":
    main()
