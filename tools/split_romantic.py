import sys

for l in open(sys.argv[1]):
    sentences = l.split('.')
    if len(sentences) == 3:
        with open('reference_input.txt', 'a') as f:
            f.write(sentences[0].strip())
            f.write('\n')
        with open(f'{sys.argv[1]}.txt', 'a') as f:
            f.write(sentences[1].strip())
            f.write('\n')
    else:
        sentences = l.split('\t')
        print(sentences)
        with open('reference_input.txt', 'a') as f:
            f.write(sentences[0].strip())
            f.write('\n')
        with open(f'{sys.argv[1]}.txt', 'a') as f:
            f.write(sentences[1].strip())
            f.write('\n')
