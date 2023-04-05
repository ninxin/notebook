
Lr = [['a', 'd', 'cd'], ['de'], ['e', 'f']]
len(Lr[1])

def solution(Input, k):
    lr = []
    max = 1
    for i in range(k):
        max *= len(Input[i])

    while len(lr) < max:

        for i in range(k):
            st = ''

