import matplotlib.pyplot as plt

font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 20,
         }

fig = plt.figure(figsize=(10, 10))
sub = fig.add_subplot(111)

X = [0.01, 0.1, 1, 10]
Y_acc = [98.41, 98.05, 97.48, 96.83]
Y_prec = [97.65, 96.85, 95.59, 94.46]
Y_rec = [98.21, 98.08, 97.92, 97.42]
Y_f1 = [97.93, 97.46, 96.74, 95.92]

l_acc, = sub.semilogx(X, Y_acc, '^-', linewidth=3, ms=10)
l_prec, = sub.semilogx(X, Y_prec, '^-', linewidth=3, ms=10)
l_rec, = sub.semilogx(X, Y_rec, '^-', linewidth=3, ms=10)
l_f1, = sub.semilogx(X, Y_f1, '^-', linewidth=3, ms=10)

# fig, ax = plt.subplots()
# ax.set_xscale("log")
# plt.semilogx(Y_acc)

plt.grid()
plt.tick_params(labelsize=15)

plt.xlabel("Alpha", font1)
plt.ylabel("Metric.(%)", font1)

plt.legend(handles=[l_acc, l_prec, l_rec, l_f1], labels=['Accuracy', 'Precision', 'Recall', 'F1'], prop=font1)
plt.savefig("images/aplha.png")
