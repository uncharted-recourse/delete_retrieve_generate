preds_a = 'delete_only_sentiment/preds.49'
preds_b = 'working_dir_sentiment/preds.49'
inputs = 'working_dir_sentiment/inputs.49'
golds = 'working_dir_sentiment/golds.49'

for l, g, p_a, p_b in zip(open(inputs), open(golds), open(preds_a), open(preds_b)):
    if '<unk>' in l:
        continue
    else:
        print("%-30s %s\n" % ('Input text:', l))
        print("%-30s %s\n" % ('Target text:', g))
        print("%-30s %s\n" % ('Delete only model:', p_a))
        print("%-30s %s\n" % ('Delete and retrieve model:', p_b))
