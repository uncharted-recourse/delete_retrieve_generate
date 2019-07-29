import sys


for l in open(sys.argv[0]):
    sentences = l.split('.'):
    with open('reference_input.txt', 'a') as f:
        f.write(sentences[0].strip())
    with open(f'{sys.argv[0]}.txt', 'a') as f:
        f.write(sentences[1].strip()


