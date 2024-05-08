
count = 0
total = 0

with open('results/datasets/imvia/Home_01/results_3.txt', 'r') as file:
  lines = file.readlines()

for line in lines:
  total += float(line.split()[3][:-2])
  count += 1
  
print(total / count)