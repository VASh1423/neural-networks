import numpy as np

def act(x):
    return 0 if x<0.5 else 1

def go (house, rock, attr):
    x = np.array([house, rock, attr])
    w11 = [0.3, 0.3, 0]
    w22 = [0.4, -0.5, 1]
    weight1 = np.array([w11, w22])
    weight12 = np.array([-1, 1])

    sum_hidden = np.dot(weight1, x)
    print("Сумма на нейронах скрытого слоя: " + str(sum_hidden))

    out_hidden = np.array([act(x) for x in sum_hidden])
    print("Значения выходного слоя: " + str(out_hidden))

    sum_end = np.dot(weight12, out_hidden)
    y = act(sum_end)
    print("Выходные значения НН: " + str(y))

    return y

res = go(1, 0, 1)
if res == 1:
    print("Ты мне нравишься")
else:
    print("Ты мне не нравишься")