import matplotlib.pyplot as plt

font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 20,
         }

fig = plt.figure(figsize=(10, 10))
sub = fig.add_subplot(111)

X = [0.05, 0.2, 0.5, 0.7, 1]
Y_avg = [95.83, 97.00, 97.68, 97.91, 98.05]
Y_max = [96.13, 97.25, 97.96, 98.16, 98.40]
Y_min = [95.45, 96.76, 97.36, 97.62, 97.78]

l_avg, = sub.plot(X, Y_avg, '^-', linewidth=3, ms=10)
l_max, = sub.plot(X, Y_max, '^-', linewidth=3, ms=10)
l_min, = sub.plot(X, Y_min, '^-', linewidth=3, ms=10)


plt.grid()
plt.tick_params(labelsize=15)

plt.xlabel("Training Set Size", font1)
plt.ylabel("Acc.(%)", font1)

plt.legend(handles=[l_avg, l_max, l_min], labels=['Average', 'Max', 'Min'], prop=font1)
plt.savefig("images/size.png")
