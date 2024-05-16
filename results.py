
directory = 'Home_02'

def accuracy(tp, fp, tn, fn):
  return ((tp + tn)/(tp + tn + fp + fn)) * 100

def sensitivity(tp, fn):
  try:
    return (tp/(tp + fn)) * 100
  except ZeroDivisionError:
    return 0
  
def specificity(fp, tn):
  return (tn/(tn + fp)) * 100
  

with open(f'results/datasets/imvia/{directory}/results_1.txt', 'r') as file:
  lines = file.readlines()

acc = 0
sens = 0
spec = 0
count = 0
time = 0

for line in lines:
  data = line.split(' & ', )
  time += float(data[2])
  count = count + 1 
  
  TP = int(data[3])
  FP = int(data[4])
  TN = int(data[5])
  FN = int(data[6][:-4])
  
  acc += accuracy(TP, FP, TN, FN)
  sens += sensitivity(TP, FN)
  spec += specificity(FP, TN)
  


print(f'{directory} & {round(acc/count, 2)} & {round(sens/count, 2)} & {round(spec/count, 2)} \\\\')
# print(total / count)

