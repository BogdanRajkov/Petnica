import matplotlib.pyplot as plt

avg, mini = [], []
f = open('fitness.txt')
content = f.readlines()
f.close()
for line in content:
    space = line.index(' ')
    mini.append(float(line[:space]))
    avg.append(float(line[space+1:]))
plt.plot(avg, label='Mediana fitnesa')
plt.plot(mini, label='Najnizi fitnes')
plt.xlabel('Broj generacija')
plt.ylabel('Fitnes')
plt.legend()
plt.title('Grafik zavisnosti fitnesa od broja generacije')
plt.show()
