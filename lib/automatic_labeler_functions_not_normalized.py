import stumpy
import os
import pandas as pd
import numpy as np
import ast
import re

#from stumpy import config
#config.STUMPY_EXCL_ZONE_DENOM = 1

# Class to manage the searches
class Comparing:
    def __init__(self, Q_df, T_df, distance_threshold):
        self.Q_df = Q_df # Seed
        self.T_df = T_df # Entire Video
        self.matches_idxs = [] # Store the matches with distance threshold equals to this: max(np.mean(D) - self.distance_threshold * np.std(D), np.min(D))
        self.all_matches_idxs = [] # Store the matches with distance threshold equals to 9e100
        self.all_mass_idxs = [] # Store all the matches (Using mass function to do this because the match function do not get all the possible matches\)
        self.filter_matches_idxs = [] # Indexes selected after the matching process
        self.measure_name = None
        self.distance_threshold = distance_threshold

    def calc_matches(self):
        # Garantir que Q_df e T_df sejam 1-dimensionais
        if isinstance(self.Q_df, pd.DataFrame):
            if self.Q_df.shape[1] != 1:
                raise ValueError("Q_df deve conter apenas uma coluna para a análise com stumpy.mass.")
            self.Q_df = self.Q_df.iloc[:, 0]  # Converter para Série
        if isinstance(self.T_df, pd.DataFrame):
            if self.T_df.shape[1] != 1:
                raise ValueError("T_df deve conter apenas uma coluna para a análise com stumpy.mass.")
            self.T_df = self.T_df.iloc[:, 0]  # Converter para Série

        # Validar os dados
        if self.Q_df.empty or self.T_df.empty:
            raise ValueError("Q_df ou T_df estão vazios. Verifique os dados de entrada.")
        if self.Q_df.isnull().values.any() or self.T_df.isnull().values.any():
            raise ValueError("Q_df ou T_df contêm valores NaN. Remova ou substitua os valores ausentes.")

        # Depuração: Verificar as dimensões e o conteúdo das séries
        print(f"Shape de Q_df: {self.Q_df.shape}, Shape de T_df: {self.T_df.shape}")
        print(f"Q_df head:\n{self.Q_df.head()}")
        print(f"T_df head:\n{self.T_df.head()}")

        # Ajustar max_distance para evitar valores negativos
        try:
            self.matches_idxs = stumpy.match(
                self.Q_df,
                self.T_df,
                max_distance=lambda D: max(np.mean(D) - self.distance_threshold * np.std(D), 0)
            )

            self.all_matches_idxs = stumpy.match(
                self.Q_df,
                self.T_df,
                max_distance=lambda D: max(9e100, np.min(D))
            )
            
            self.all_mass_idxs = stumpy.mass(self.Q_df, self.T_df)

        except ValueError as e:
            print(f"Erro durante o cálculo: {e}")
            raise





def get_euclidean_distance(target, matrix):
    for subarray in matrix:
        if target in subarray:
            return subarray[0]
    return None

def label_current_series(
    current_path_location, 
    RESUME_DT, 
    selected_measures_in_frame_interval, 
    dict_label_parameters, 
    seed_name, 
    LABELED_FILE_NAME='VD_LABELED_L0.CSV', 
    distance_threshold=2, 
    frame_threshold=3
):
    VD_MEASURE_DT = pd.read_csv(current_path_location)
    
    if 'Unnamed: 0' in VD_MEASURE_DT.columns:
        VD_MEASURE_DT.drop(columns=['Unnamed: 0'], inplace=True)

    T_df = VD_MEASURE_DT[dict_label_parameters['reference_measures']]

    # Apply Stumpy functions
    object_list = []
    temp_row = pd.DataFrame()

    matches_memory = []
    all_matches_memory = []
    all_mass_memory = []
    
    # Certifique-se de que selected_measures_in_frame_interval seja um DataFrame
    selected_measures_in_frame_interval = pd.DataFrame(selected_measures_in_frame_interval)

    # Iterar pelas colunas da série de referência
    for step, column in enumerate(selected_measures_in_frame_interval.columns):
        comp_object = Comparing(
            selected_measures_in_frame_interval[column],
            T_df[column],
            distance_threshold
        )
        comp_object.calc_matches()

        matches_memory.append(comp_object.matches_idxs)
        all_mass_memory.append(comp_object.all_mass_idxs)

    # Apply the matching filter
    all_index = []
    for c in object_list:  
        all_index.append(c.matches_idxs[:, 1])
    
    # Aplicar o filtro por coincidência de índices
    aux = all_index.copy()

    # Verificar se aux está vazio
    if not aux:
        print(f"Nenhuma correspondência inicial encontrada em {current_path_location}.")
        return None  # Retorna None em vez de usar continue

    # Chamar find_all_matches com verificação
    filter_index = find_all_matches(aux, frame_threshold)

    # Verificar se filter_index está vazio
    if not filter_index or not isinstance(filter_index, list) or len(filter_index) == 0 or not filter_index[0]:
        print(f"Nenhuma correspondência encontrada em filter_index para {current_path_location}.")
        return None  # Retorna None em vez de usar continue

    # Se chegamos aqui, filter_index_list está definido
    filter_index_list = list(filter_index[0])

    # Processar os índices encontrados
    filter_index_begin = []
    for idx_tuple in filter_index_list:
        filter_index_begin.append(idx_tuple)

    idxs_match_frame_seq = []
    for idx in filter_index_begin:
        idx_frame_seq = VD_MEASURE_DT.loc[idx, 'frame_seq']
        for c in object_list:
            ed = get_euclidean_distance(idx, c.matches_idxs)
            if ed is not None:
                break
        idxs_match_frame_seq.append([idx_frame_seq, ed])

    # Test if the Labeled File was already created
    VD_LABEL_PATH = (os.path.join(os.path.dirname(current_path_location), LABELED_FILE_NAME))
    test = os.path.exists(VD_LABEL_PATH)

    if test:
        VD_LABELED_DT = pd.read_csv(VD_LABEL_PATH)

        if 'Unnamed: 0' in VD_LABELED_DT.columns:
            VD_LABELED_DT.drop(columns=['Unnamed: 0'], inplace=True)

        VD_LABELED_DT = VD_LABELED_DT.set_index(pd.Index(VD_LABELED_DT['frame_seq']))

    else:
        # First Initiate the labels = 0 means NO Label
        VD_LABELED_DT = VD_MEASURE_DT.copy()
        VD_LABELED_DT['label_measures'] = str({})
        VD_LABELED_DT = VD_LABELED_DT.set_index(pd.Index(VD_LABELED_DT['frame_seq']))
    
    temp_row['final'] = (len(filter_index[0]))

    gap_occurrences = 0
    occurrences_len = []
    
    # Adds information to label the frames.
    for label_idx in idxs_match_frame_seq:
        init_lab = label_idx[0]
        end_lab = init_lab + len(selected_measures_in_frame_interval) - 1
        e_distance = label_idx[1]
        FRAMES_DT = VD_LABELED_DT.query(f'frame_seq >= {init_lab} & frame_seq <= {end_lab}')
        occurrences_len.append(len(FRAMES_DT))
        
        # if there is not a discontinuity in the interval of frames
        if not end_lab in VD_LABELED_DT.index:
            
            # In cases that the missing frames are in the end of interval
            VD_LABELED_DT, removed_series = UPDATE_LABEL_DF(
                init_lab, 
                end_lab, 
                dict_label_parameters['label_name'], 
                dict_label_parameters['reference_measures'], 
                VD_LABELED_DT, 
                seed_name, 
                e_distance
            )
            temp_row['final'] -= removed_series
        else:
            gap_occurrences += 1

    temp_row['final'] -= gap_occurrences
    RESUME_DT = pd.concat([RESUME_DT, temp_row], axis=0)
    
    # Save CSV file
    VD_LABELED_DT.reset_index(drop=True, inplace=True)
    VD_LABELED_DT.to_csv(VD_LABEL_PATH)

    return RESUME_DT, matches_memory, all_matches_memory, all_mass_memory, idxs_match_frame_seq, occurrences_len


def UPDATE_LABEL_DF(init_lab, end_lab, label_name_in, label_measure_in, data_frame_in, seed_name, matches_idxs):
    # Extract subset from 'label_measures' for specified interval
    subset_df = data_frame_in.loc[init_lab:end_lab, 'label_measures']

    # Convert strings to dictionaries and filter out any empty dictionaries (`{}`)
    non_empty_label_measures = subset_df[subset_df != '{}']
    
    # Convert the dict into a list
    non_empty_label_measures_list = list(non_empty_label_measures)
    
    # Convert the list into a set to remove duplicated items
    unique_label_measures = set(non_empty_label_measures_list)

    # Re-convert the set into a list to be able to iterate it
    unique_label_measures_list = list(unique_label_measures)
    
    # Verify if all the euclidean distances on the current dataset are bigger than the euclidean distance arriving
    # If there is any euclidean distance smaller than the occurrence arriving, it WONT be saved.
    for label_measure in unique_label_measures_list:
        label_measure_to_str = str(label_measure)

        label_measure_dict = ast.literal_eval(label_measure_to_str)
        euclidean_distance = list(label_measure_dict.values())[0][1]

        if matches_idxs > euclidean_distance:
            return data_frame_in, 1 # Returns 1 to remove the matches_idxs occurrence, since it was not marked.

    # Amount of series that were cleaned after the process (Used to decrement the amount of occurrences of that singular seed)
    removed_series = 0

    # Replace all cells in `data_frame_in` that match any unique dictionary with `'{}'`
    for label_measure in unique_label_measures_list:
        label_measure_to_str = str(label_measure)

        label_measure_dict = ast.literal_eval(label_measure_to_str)

        removed_series += 1
        data_frame_in['label_measures'] = data_frame_in['label_measures'].apply(
            lambda x: '{}' if x == label_measure_to_str else x
        )

    # Extract video number from 'seed_name'
    match = re.search(r'VD_R_(\d+)', seed_name)
    video_num = int(match.group(1)) if match else seed_name

    # Define the new value as a dictionary and then convert to a string
    new_label_measure = str({label_name_in: (label_measure_in, matches_idxs, video_num)})
    
    # Update rows from `init_lab` to `end_lab` with `new_value` in 'label_measures'
    data_frame_in.loc[init_lab:end_lab, 'label_measures'] = data_frame_in.loc[init_lab:end_lab, 'label_measures'].apply(
        lambda x: str({**ast.literal_eval(x), **ast.literal_eval(new_label_measure)}) if x != '{}' else new_label_measure
    )

    return data_frame_in, removed_series

def find_close_values(idxs, threshold):
    close_values = []

    # Compare each pair of lists
    for i in range(len(idxs)):
        for j in range(i+1, len(idxs)):
            list1 = idxs[i]
            list2 = idxs[j]

            # Compare every element of both lists
            for num1 in list1:
                for num2 in list2:

                    # If the distance between the values are smaller than the threshold, consider it accepted.
                    if abs(num1 - num2) <= threshold:
                        close_values.append(min(num1,num2))
    close_values = set(close_values)
    return close_values

def find_all_matches(list_of_index, threshold):
    if not list_of_index or len(list_of_index) <= 1:
        print("Poucos índices para calcular correspondências. Retornando lista vazia.")
        return []
    else:
        list_aux = []
        list_aux.append(list_of_index.pop(0))
        list_aux.append(list_of_index.pop(0))
        result = find_close_values(list_aux, threshold)
        list_of_index.insert(0, result)
        return find_all_matches(list_of_index, threshold)

