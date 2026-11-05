import sys
!{sys.executable} -m pip install scikit-fuzzy
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
temperature=ctrl.Antecedent(np.arange(0,41,1),'temperature')
fan_speed=ctrl.Consequent(np.arange(0,101,1),'fan speed')
temperature['low']=fuzz.trimf(temperature.universe,[0,0,20])
temperature['medium']=fuzz.trimf(temperature.universe,[10,20,30])
temperature['high']=fuzz.trimf(temperature.universe,[20,40,60])
fan_speed['low']=fuzz.trimf(fan_speed.universe,[0,0,50])
fan_speed['medium']=fuzz.trimf(fan_speed.universe,[25,50,75])
fan_speed['high']=fuzz.trimf(fan_speed.universe,[50,100,100])
rule1=ctrl.Rule(temperature['low'],fan_speed['low'])
rule2=ctrl.Rule(temperature['medium'],fan_speed['medium'])
rule3=ctrl.Rule(temperature['high'],fan_speed['high'])
fan_control=ctrl.ControlSystem([rule1,rule2,rule3])
fan_simulation=ctrl.ControlSystemSimulation(fan_control)
#test temperature
temp=35
fan_simulation.input['temperature']=temp
fan_simulation.compute()
print("temperature",temp,"c")
print("fan speed",round(fan_simulation.output['fan speed'],2),"%")
temperature.view()
fan_speed.view()
plt.show()
