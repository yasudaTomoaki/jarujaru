from transformers import T5Tokenizer,T5ForConditionalGeneration
def generate_text_from_model(word, max_length_src = 5, max_length_target = 100, num_return_sequences = 3):
    model_path = 'jarujaru_model'
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    train_model = T5ForConditionalGeneration.from_pretrained(model_path)
    train_model.eval()
    batch = tokenizer([word], max_length = max_length_src, truncation = True, padding = 'longest', return_tensors = 'pt')

    # 生成処理を行う
    outputs = train_model.generate(input_ids = batch['input_ids'], attention_mask = batch['attention_mask'], max_length = max_length_target,
                                    repetition_penalty = 8.0, num_return_sequences = num_return_sequences)

    generated_texts = [tokenizer.decode(ids, skip_special_tokens = True, clean_up_tokenization_spaces = False) for ids in outputs]
    return generated_texts

word = input()
title = generate_text_from_model(word)
for t in title:
    print(t)