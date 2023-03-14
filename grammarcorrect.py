from gramformer import Gramformer
import torch

def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(1212)

gf = Gramformer(models=1, use_gpu=False) # 1=corrector, 2=detector

fi_incorrect = open("incorrect.txt", "r").read()
fi_correct = open("corrected.txt", "w")

influent_sentences = [
    fi_incorrect
]   

for influent_sentence in influent_sentences:
    corrected_sentences = gf.correct(influent_sentence, max_candidates=1)
    print("[Input] ", influent_sentence)
    print ("[Input] ", influent_sentence, file = fi_correct)
    for corrected_sentence in corrected_sentences:
        print("[Corrected] ",corrected_sentence)
        print ("[Corrected] ",corrected_sentence, file = fi_correct)
    print("-" *100)

fi_correct.close()