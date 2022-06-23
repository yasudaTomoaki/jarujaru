import pickle
from torch.utils.data import DataLoader
from transformers import T5Tokenizer
from sklearn.model_selection import train_test_split

class Jarujaru_Load:
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained("sonoisa/t5-base-japanese",is_fast=True)
        self.enc_max_len = 3
        self.dec_max_len = 50
        self.batch_size=16
    
    def load_data(self):
        with open('jarujaru/jarujaru.pkl','rb') as f:
            data = pickle.load(f)
        input = list(data.keys())
        output = list(data.values())
        x_train,x_test,t_train,t_test=train_test_split(input,output,test_size=0.2, random_state=42, shuffle=True)
        train_data = [(src, tgt) for src, tgt in zip(x_train, t_train)]
        test_data = [(src, tgt) for src, tgt in zip(x_test, t_test)]
        train,test=self.convert_batch_data(train_data,test_data)
        return train,test

    
    def convert_batch_data(self,train_data, valid_data):
        enc_max_len = self.enc_max_len
        dec_max_len = self.dec_max_len
        tokenizer=self.tokenizer
        def generate_batch(data):
            batch_src, batch_tgt = [], []
            for src, tgt in data:
                batch_src.append(src)
                batch_tgt.append(tgt)

            batch_src = tokenizer(batch_src, max_length=enc_max_len, truncation=True, padding="max_length", return_tensors="pt")
            batch_tgt = tokenizer(batch_tgt, max_length=dec_max_len, truncation=True, padding="max_length", return_tensors="pt")

            return batch_src, batch_tgt

        train_iter = DataLoader(train_data, batch_size=self.batch_size, collate_fn=generate_batch)
        valid_iter = DataLoader(valid_data, batch_size=self.batch_size, collate_fn=generate_batch)

        return train_iter, valid_iter
                    