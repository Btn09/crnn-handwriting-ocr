from spellchecker import SpellChecker
spell = SpellChecker()

def ngram_correction(prev_word, current_pred, bigram_counts):
    candidates = spell.candidates(current_pred)
    if not candidates or not prev_word:
        return spell.correction(current_pred)
    
    best_word = current_pred
    max_prob = -1
    
    for cand in candidates:
        prob = bigram_counts.get((prev_word.lower(), cand.lower()), 0)
        
        if prob > max_prob:
            max_prob = prob
            best_word = cand
            
    return best_word