import matplotlib.pyplot as plt
import json

with open("./ResNet18_results.json", 'r') as j:
    contents = json.loads(j.read())

results_list = []
name_list = []
sev_list = []
for value in sorted(contents.iterkeys()):
    if str(value) != "clean":
        if int(str(value).split()[1]) != 5:
            sev_list.append(contents[value])
        else:
            sev_list.append(contents[value])
            results_list.append(sev_list)
            sev_list = []
            name_list.append(str(value).split()[0])

results_list.append([contents["clean"], contents["clean"], contents["clean"],contents["clean"], contents["clean"]])
name_list.append("clean")

fig = plt.figure(figsize=(8, 6), dpi=70)
plt.xlabel('Severity', fontsize=12)
x = [1, 2, 3, 4, 5]
default_x_ticks = range(len(x))
plt.ylabel("Accuracy", fontsize=12)
for i in range(len(results_list)):
    if name_list[i] == "clean":
        plt.plot(default_x_ticks, results_list[i], label=name_list[i], linestyle='--')
    else:
        plt.plot(default_x_ticks, results_list[i], label=name_list[i], marker='o')
    plt.xticks(default_x_ticks, x)
plt.legend()
plt.show()
plt.savefig('Augmentations.png')
