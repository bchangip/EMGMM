from random import randint

with open('generatedPoints.txt', 'w') as outputFile:
  for i in range(1000):
    # outputFile.write('['+str(randint(-500,500))+', '+str(randint(-500,500))+']\n')
    outputFile.write('['+str(randint(-500,-300))+', '+str(randint(-450,-300))+']\n')
    outputFile.write('['+str(randint(100,300))+', '+str(randint(200,500))+']\n')
    outputFile.write('['+str(randint(-100,0))+', '+str(randint(0,100))+']\n')