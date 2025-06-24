import json
import re
import torch
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Download sentence tokenizer
nltk.download('punkt')

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
checkpoint = "facebook/nllb-200-1.3B"
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Language code mapping
LANG_CODES = {
    'de': 'deu_Latn',  # German
    'ja': 'jpn_Jpan',  # Japanese
    'ru': 'rus_Cyrl',  # Russian
    'sw': 'swh_Latn',  # Swahili
    'th': 'tha_Thai'   # Thai
}

# Mapping to NLTK language codes
NLTK_LANG_MAP = {
    'de': 'german',
    'ja': 'english',   # fallback, Japanese not supported
    'ru': 'russian',
    'sw': 'english',   # fallback
    'th': 'english'    # fallback
}

def split_text_by_language(text, lang, max_chunk_length=200):
    """Split text into sentence-based chunks based on language."""
    nltk_lang = NLTK_LANG_MAP.get(lang, 'english')

    try:
        sentences = sent_tokenize(text, language=nltk_lang)
    except:
        # Fallback to splitting by newline or punctuation
        sentences = re.split(r'(?<=[\.\n])\s+', text.strip())

    # Combine into chunks
    chunks = []
    buffer = ''
    for sentence in sentences:
        if len(buffer) + len(sentence) < max_chunk_length:
            buffer += sentence + ' '
        else:
            chunks.append(buffer.strip())
            buffer = sentence + ' '
    if buffer.strip():
        chunks.append(buffer.strip())
    return chunks

def translate_text(text, lang):
    """Translate text from a given language to English using NLLB."""
    if not text.strip():
        return ""

    source_lang_code = LANG_CODES[lang]
    tokenizer.src_lang = source_lang_code
    text = text.replace("</s>", "").replace("<eos>", "").strip()

    chunks = split_text_by_language(text, lang, max_chunk_length=200)
    translations = []

    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True).to(device)
        bos_token_id = tokenizer.convert_tokens_to_ids("eng_Latn")

        outputs = model.generate(
            **inputs,
            forced_bos_token_id=bos_token_id,
            max_length=512,
            num_beams=4,
            repetition_penalty=2.5,
            no_repeat_ngram_size=4,
            length_penalty=0.9,
            early_stopping=True
        )
        translated_chunk = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        translations.append(translated_chunk.strip())

    return ' '.join(translations)

def process_file():
    # Load the extracted questions
    with open('extracted_questions3.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Process each question
    for question in tqdm(data, desc="Translating questions"):
        for lang, lang_data in question.items():
            if lang == 'question_number':
                continue

            if 'model_solution' in lang_data and lang != 'en':
                try:
                    translated = translate_text(lang_data['model_solution'], lang)
                    lang_data['english_translation'] = translated
                except Exception as e:
                    print(f"Translation error in language '{lang}' for question {question.get('question_number')}: {e}")
                    lang_data['english_translation'] = "[Translation failed]"

    # Save results
    with open('extracted_questions_translated2.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("\n✅ Translation complete. Results saved to 'extracted_questions_translated2.json'.")

if __name__ == "__main__":
    process_file()
