import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



data = pd.read_csv('salary.csv')




def linear_regression(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    
    n = len(points)
    for i in range(n):
        x = points.iloc[i].YearsExperience
        y = points.iloc[i].Salary
        
        m_gradient += -(2/n) * x *(y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))
        
    m = m_now - m_gradient * L
    b  = b_now - b_gradient * L
    return m, b

m = 0
b = 0
L = 0.001

for i in range(500):
    m, b = linear_regression(m, b, data, L)
print(m, b)

user = input("Enter years of experience to predict salary: ")

user_exp = float(user)
salary = m * user_exp + b
print(f"Your expected salary : ${salary:.2f}")


