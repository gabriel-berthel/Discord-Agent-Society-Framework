import pickle
with open('./qa_bench/qa_bench_histo.pkl', 'rb') as f:
    historic = pickle.load(f)
    
with open ('historic.txt', 'w') as f:
    for l in historic:
        f.writelines(f'{l}\n')