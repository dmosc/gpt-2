from wakepy import keep
from train_tokenizer import train_tokenizer
from train_model import train_model


if __name__ == '__main__':
    with keep.presenting():
        train_tokenizer()
        train_model()
