

def split_text_to_files():
    with open('test.txt') as file:
        text = file.readlines()

    print(text)


split_text_to_files()