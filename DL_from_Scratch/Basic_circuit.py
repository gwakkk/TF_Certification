import numpy as numpy

# perceptron
# 다수의 신호를 입력으로 받아 하나의 신호를 출력하는 것
# 간단한 기본 게이트 AND / OR / XOR / NAND 구현

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def NAND(x1, x2):
# NAND는 AND와 가중치만 다르다
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1   

# AND NAND OR은 모두 같은 구조의 퍼셉트론이고, 가중치 매개변수의 값만 다르다


# XOR은 1차 선형분리가 불가능
# XOR은 NAND와 OR을 결합한 형태를 AND로 연산하면 결과가 동일하게 나오게 된다.
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y
