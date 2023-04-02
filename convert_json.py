import codecs
import json


lines = []
with open('benchmark_dataset/The Python 100.txt', mode="r", encoding="utf-8") as corpus_file:
    for line in corpus_file:
        if line.strip():
            lines.append(line.strip())

qa = {}
passage = []
question = []
answer = []
distractor = []
temp = []
index = 0
current_concept = ''
qa_no = 1

while index < len(lines):
    line = (lines[index].encode('ascii', 'ignore')).decode('utf-8')
    if line.startswith('Concept:'):
        current_concept = line
        qa[current_concept] = []
        index += 1

    indecent = len(str(qa_no)) + 2
    p = (lines[index].encode('ascii', 'ignore')).decode('utf-8')[indecent:]
    q = (lines[index+1].encode('ascii', 'ignore')).decode('utf-8')
    a = (lines[index+2].encode('ascii', 'ignore')).decode('utf-8')[indecent:]
    d = [(l.encode('ascii', 'ignore')).decode('utf-8')[indecent:] for l in lines[index+3:index+6]]
    print(qa_no)
    print({'passage': p, 'question': q, 'answer': a, 'distractor': d})
    qa[current_concept].append({'passage': p, 'question': q, 'answer': a, 'distractor': d,
                                'answer_index': p.lower().find(a.lower())})
    index += 7
    qa_no += 1

with open('benchmark_dataset/python_programming.json', 'w') as f:
    json.dump(qa, f, indent=1)
