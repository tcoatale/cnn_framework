#%%
import numpy as np
import glob
import pandas as pd 
import pickle

dir_path = 'raw/partial_nephrectomy/preprocessed/'
dest_path = 'data/partial_nephrectomy/'
images_per_sec = 6
images_per_file = 10000

# 135 * 480 * 3
#%%
def convert_time_to_index(time):
    '''
    Converts a datetime.time object to seconds , then convert the seconds to an index, based on a conversion rate
    '''
    secs = time.second
    mins = time.minute
    hours = time.hour
    
    total_secs = secs + 60 * mins + 3600 * hours
    
    return images_per_sec * total_secs
    
def get_segmentation_dataframe(file_path):
    segmentation_df = pd.read_excel(file_path)    
    segmentation_df.Start = segmentation_df.Start.apply(convert_time_to_index)
    segmentation_df.Finish = segmentation_df.Finish.apply(convert_time_to_index)
    segmentation_df['n_frames'] = segmentation_df.Finish - segmentation_df.Start
    segmentation_df['true_start'] = segmentation_df.n_frames.cumsum() - segmentation_df.n_frames
    segmentation_df['true_end'] = segmentation_df.true_start + segmentation_df.n_frames
    
    return segmentation_df
    
def convert_to_vector_data(indices, labels, images):
    images_and_labels = zip(indices, labels.tolist(), images.tolist())
    
    vector_data = list(map(lambda label_image: [label_image[0]] + label_image[1]  + label_image[2], images_and_labels))
    vector_data = np.array(vector_data, dtype=np.uint8)
    
    return vector_data

def get_vectorized_data(reverse_labels, df_line):
    file = df_line.Filename

    print(df_line.true_start, '-', df_line.true_end, end=" ")

    with open(dir_path + file + '.pkl', 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        frames = u.load()
        
    line_indices = list(range(df_line.true_start, df_line.true_end))
    line_images = frames[df_line.Start:df_line.Finish]
    line_images = line_images.reshape((line_images.shape[0], -1))
    line_labels = np.array([[reverse_labels[df_line.Description]]] * line_images.shape[0], dtype='uint8')

    vector_data = convert_to_vector_data(line_indices, line_labels, line_images)
    
    print('Done.')

    return [vector_data]

def write_chunk(chunks, file_path, index):
    binary_data = chunks[index].reshape(-1)
    binary_data = bytearray(binary_data)
    
    newFile = open (file_path + '_' + str(index), "wb")
    newFile.write(binary_data)
    
    return index

def preprocess_data(file_path):
    segmentation_df = get_segmentation_dataframe(file_path)
    labels = segmentation_df.Description.unique().tolist()
    reverse_labels = dict(list(map(lambda l: (l, labels.index(l)), labels)))

    individual_video_data = segmentation_df.apply(lambda l: get_vectorized_data(reverse_labels, l), axis=1).tolist()
    individual_video_data = list(map(lambda d: d[0], individual_video_data))
    full_data = np.vstack(individual_video_data)

    chunks = [full_data[i:i+images_per_file] for i in range(0, full_data.shape[0], images_per_file)]

    list(map(lambda i: write_chunk(chunks, dest_path + file_path.split('/')[-1][:-5] + '_batch', i), range(len(chunks))))
    
#%%
segmentations = glob.glob(dir_path + '*T*.xlsx')
file_path = segmentations[0]
preprocess_data(file_path)