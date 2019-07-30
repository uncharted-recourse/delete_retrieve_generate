import sys
#preds_a = 'delete_only_sentiment/preds.49'
preds_a = 'romantic_humorous/preds.246'
#preds_b = 'working_dir_sentiment/preds.49'
inputs = 'data/imagecaption/reference_input.txt'
inputs_context = 'romantic_humorous/inputs.246'
golds = 'romantic_humorous/golds.246'
#golds = 'data/imagecaption/reference.0.txt'
#inputs = 'working_dir_sentiment/inputs.49'
#golds = 'working_dir_sentiment/golds.49'

count=0
for l, g, p_a, i_c in zip(open(inputs), open(golds), open(preds_a), open(inputs_context)):
    if count == 10:
        break
    if '<unk>' in g or '<unk>' in l:
        continue
    else:
        sys.stdout.write("%-30s %s" % ('Input text:', l))
        #sys.stdout.write("%-30s %s" % ('Input context:', i_c))
        sys.stdout.write("%-30s %s" % ('Target text:', g))
        sys.stdout.write("%-30s %s\n" % ('Output text:', p_a))
        #sys.stdout.write("%-30s %s" % ('Delete only model:', p_a))
        #sys.stdout.write("%-30s %s\n" % ('Delete and retrieve model:', p_b))
        count+=1
