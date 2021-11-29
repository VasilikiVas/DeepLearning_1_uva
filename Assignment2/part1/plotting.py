import matplotlib.pyplot as plt
import json

with open(results_filename, 'r') as j:
    contents = json.loads(j.read())

fig = plt.figure(figsize=(8, 6), dpi=70)
plt.xlabel('Number of Epochs', fontsize=12)
plt.ylabel(name, fontsize=12)
plt.plot(list_train, label="Training Set")
plt.plot(list_valid, label="Validation Set")
plt.legend()
plt.show()
