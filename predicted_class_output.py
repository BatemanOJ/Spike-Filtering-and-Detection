
# Counts the number predicted in each class
def predicted_class_output(predicted_class_output):
    counter1=0
    counter2=0
    counter3=0 
    counter4=0
    counter5=0
    for i in range(len(predicted_class_output)):
        if predicted_class_output[i] == 1:
            counter1 = counter1 + 1
        elif predicted_class_output[i] == 2:
            counter2 = counter2 + 1
        elif predicted_class_output[i] == 3:
            counter3 = counter3 + 1
        elif predicted_class_output[i] == 4:
            counter4 = counter4 + 1
        elif predicted_class_output[i] == 5:
            counter5 = counter5 + 1
    counters = [counter1, counter2, counter3, counter4, counter5]
    return counters