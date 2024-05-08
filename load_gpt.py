import os
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "/models/" 
model = AutoModelForCausalLM.from_pretrained(path)
tokenizer = AutoTokenizer.from_pretrained(path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

input_prompt = "You are an professional colorectal cancer specialist, can you provide a personal treatment plan for following patient:  Mr. Benton is a 67-year-old gentleman with stage T3a N0 M0 rectal adenocarcinoma status post neoadjuvant chemoradiation followed by low anterior resection of his rectal cancer.  He had an allergic response to propofol, muscle relaxant and cefotetan as part of his preop on March 11th leading to hypertension, bronchospasm and cardiac arrest requiring CPR.  He had a definitive surgery on February 1, 2011. On pathology, Mr. Benton's tumor was moderately differentiated rectal adenocarcinoma.  The lesion 3.0 cm in greatest dimension.  Invasive tumor invades through muscularis propria proximal distal and radial margin free of tumor.  Zero out of 15 lymph nodes were involved.  Mr. Benton presents today for follow-up reporting that he is recovering well with good appetite and energy level.  He has normal stool without any blood through his ostomy. He denies any nausea, vomiting or any localizing pain or symptoms.  His wound is still healing by secondary intention which has improved significantly.  REVIEW OF SYSTEMS:  Comprehensive review of systems is otherwise negative.  PAST MEDICAL HISTORY, SOCIAL HISTORY, FAMILY:  Unchanged.  PHYSICAL EXAM:  Mr. Benton appeared in no acute distress.  HEENT: Normocephalic, atraumatic.  Sclerae anicteric.  Oropharynx clear.  Neck is supple.  Lungs were clear to auscultation percussion.  Heart:  Regular rate and rhythm, S1, S2.  No murmur.  Abdomen:  Soft, nondistended, nontender. No hepatosplenomegaly.  Extremities:  No clubbing, cyanosis or edema. Neurological exam was nonfocal.  Ostomy site was pink and well-perfused.  LABORATORY DATA:  WBC 6.4, hemoglobin 9.2, hematocrit 28.5, platelet 242 from February 6th.    TREATMENT PLAN:  "
inputs = tokenizer(input_prompt, return_tensors="pt", truncation=True, max_length=1024)
inputs = {k: v.to(device) for k, v in inputs.items()}  
generated_text = model.generate(**inputs, max_length=1024, do_sample=True)
generated_text = tokenizer.decode(generated_text[0])
print(generated_text)


