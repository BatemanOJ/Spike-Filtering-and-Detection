import scipy.io as spio
import os


# Stores the index and predicted class for each data set D2-6
def data_storage(detected_spikes_d2, detected_spikes_d3, detected_spikes_d4, detected_spikes_d5, detected_spikes_d6, 
                 predicted_class_output_d_2, predicted_class_output_d_3, predicted_class_output_d_4, predicted_class_output_d_5, predicted_class_output_d_6):
    
    # Path name
    path = r'C:\Users\ollie\OneDrive - University of Bath\4th Year\Semester 1\Computational Inteligence\Coursework C\13-12-2024'
    filename2 = 'D2.mat'
    filename3 = 'D3.mat'
    filename4 = 'D4.mat'
    filename5 = 'D5.mat'
    filename6 = 'D6.mat'

    # Joins the path and the file name 
    path_to_save2 = os.path.join(path, filename2)
    path_to_save3 = os.path.join(path, filename3)
    path_to_save4 = os.path.join(path, filename4)
    path_to_save5 = os.path.join(path, filename5)
    path_to_save6 = os.path.join(path, filename6)

    # Data saves for each data set d2-6
    output_data = {
        'Index': detected_spikes_d2,
        'Class': predicted_class_output_d_2
    }
    spio.savemat(path_to_save2, output_data) 

    output_data = {
        'Index': detected_spikes_d3,
        'Class': predicted_class_output_d_3
    }
    spio.savemat(path_to_save3, output_data) 

    output_data = {
        'Index': detected_spikes_d4,
        'Class': predicted_class_output_d_4
    }
    spio.savemat(path_to_save4, output_data) 

    output_data = {
        'Index': detected_spikes_d5,
        'Class': predicted_class_output_d_5
    }
    spio.savemat(path_to_save5, output_data) 

    output_data = {
        'Index': detected_spikes_d6,
        'Class': predicted_class_output_d_6
    }
    spio.savemat(path_to_save6, output_data) 
