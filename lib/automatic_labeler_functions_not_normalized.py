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
        print(f"Tamanho de Q_df: {len(self.Q_df)}, Tamanho de T_df: {len(self.T_df)}")
        if len(self.Q_df) < 2 or len(self.T_df) < 2:
            raise ValueError("As séries precisam conter mais de 1 elemento.")

        # Tamanho da janela
        m = min(len(self.Q_df), len(self.T_df), 11)
        if m < 2:
            raise ValueError("Tamanho da janela deve ser pelo menos 2.")

        # Matriz de perfil usando stumpy.aamp
        profile = stumpy.aamp(self.T_df, m)

        # Correspondências e índices
        self.matches_idxs = profile[:, 0]  # Índices das subsequências
        self.all_distances = profile[:, 1]  # Distâncias mínimas






def get_euclidean_distance(target, matrix):
    # Verifique se `matrix` é iterável (lista ou array)
    if isinstance(matrix, (list, np.ndarray)):
        for subarray in matrix:
            # Verifique se `subarray` também é iterável
            if isinstance(subarray, (list, np.ndarray)):
                if target in subarray:
                    return subarray[0]
            # Caso contrário, compare diretamente
            elif target == subarray:
                return subarray
    # Caso `matrix` seja escalar, compare diretamente
    elif isinstance(matrix, (float, int)):
        return matrix if target == matrix else None
    return None


def label_current_series(current_path_location, RESUME_DT, selected_measures_in_frame_interval, dict_label_parameters, seed_name, LABELED_FILE_NAME='VD_LABELED_L0.CSV', distance_threshold=2, frame_threshold=3):
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
    
    for step in range(0, len(selected_measures_in_frame_interval.columns)):
        comp_object = Comparing(selected_measures_in_frame_interval[dict_label_parameters['reference_measures'][step]], T_df[dict_label_parameters['reference_measures'][step]], distance_threshold)
        comp_object.calc_matches()

        matches_memory.append(comp_object.matches_idxs)
        all_mass_memory.append(comp_object.all_mass_idxs)
        all_matches_memory.append(comp_object.all_matches_idxs)

        comp_object.measure_name = dict_label_parameters['reference_measures'][step]
        object_list.append(comp_object)
        
        # Count the number of rows
        # Lógica para lidar com um único valor
        temp_row.at[0, dict_label_parameters['reference_measures'][step]] = 1 if comp_object.matches_idxs is not None else 0

        
    # Apply the matching filter
    all_index = []
    for c in object_list:
        if isinstance(c.matches_idxs, np.ndarray):
            all_index.append(c.matches_idxs)  # Adiciona o array 1D
        else:
            all_index.append([c.matches_idxs])  # Caso seja um valor escalar, encapsula em lista

    
    # Filter by coincidence from a distance threshold between the position of each indexes
    aux = all_index.copy()
    filter_index = find_all_matches(aux, frame_threshold)
    
    filter_index_list = list(filter_index[0])  # Gera a lista inicial

    # Inicialize `filter_index_begin` com índices válidos
    filter_index_begin = [int(idx) for idx in filter_index_list if idx in VD_MEASURE_DT.index]

    # Continue com a lógica existente
    idxs_match_frame_seq = []
    for idx in filter_index_begin:
        idx_frame_seq = VD_MEASURE_DT.loc[idx, 'frame_seq']
        for c in object_list:
            if isinstance(c.matches_idxs, (float, int)):
                ed = c.matches_idxs if idx == c.matches_idxs else None
            else:
                print(f"Tipo de c.matches_idxs: {type(c.matches_idxs)}, valor: {c.matches_idxs}")
                ed = get_euclidean_distance(idx, c.matches_idxs)
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
        end_lab = init_lab+len(selected_measures_in_frame_interval)-1
        e_distance = label_idx[1]
        FRAMES_DT = VD_LABELED_DT.query(f'frame_seq >= {init_lab} & frame_seq <= {end_lab}')
        occurrences_len.append(len(FRAMES_DT))
        
        # if there is not a discontinuity in the interval of frames
        if not FRAMES_DT.get('gap', pd.Series([0] * len(FRAMES_DT))).any() == 1 and end_lab in VD_LABELED_DT.index:
            
            # In cases that the missing frames are in the end of interval
            VD_LABELED_DT, removed_series = UPDATE_LABEL_DF(init_lab, end_lab, dict_label_parameters['label_name'], dict_label_parameters['reference_measures'], VD_LABELED_DT, seed_name, e_distance)
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
    # Extrair subconjunto de 'label_measures' para o intervalo especificado
    subset_df = data_frame_in.loc[init_lab:end_lab, 'label_measures']

    # Converter strings em dicionários e filtrar dicionários vazios (`{}`)
    non_empty_label_measures = subset_df[subset_df != '{}']
    non_empty_label_measures_list = list(non_empty_label_measures)

    # Remover duplicatas convertendo para um conjunto, depois reconverter para lista
    unique_label_measures = set(non_empty_label_measures_list)
    unique_label_measures_list = list(unique_label_measures)

    # Verificar se todas as distâncias Euclidianas no conjunto atual são maiores que a distância recebida
    # Se alguma distância for menor, a ocorrência não será salva
    for label_measure in unique_label_measures_list:
        label_measure_to_str = str(label_measure)
        label_measure_dict = ast.literal_eval(label_measure_to_str)
        euclidean_distance = list(label_measure_dict.values())[0][1]

        # Verificar valores nulos antes da comparação
        if matches_idxs is not None and euclidean_distance is not None:
            if matches_idxs > euclidean_distance:
                return data_frame_in, 1  # Retorna 1 para remover a ocorrência de matches_idxs, já que não foi marcada

    # Contador para séries removidas após o processo (usado para decrementar o número de ocorrências dessa semente)
    removed_series = 0

    # Substituir todas as células em `data_frame_in` que correspondam a qualquer dicionário único por `'{}'`
    for label_measure in unique_label_measures_list:
        label_measure_to_str = str(label_measure)

        removed_series += 1
        data_frame_in['label_measures'] = data_frame_in['label_measures'].apply(
            lambda x: '{}' if x == label_measure_to_str else x
        )

    # Extrair número do vídeo a partir de 'seed_name'
    match = re.search(r'VD_R_(\d+)', seed_name)
    video_num = int(match.group(1)) if match else seed_name

    # Definir o novo valor como um dicionário e depois converter para string
    new_label_measure = str({label_name_in: (label_measure_in, matches_idxs, video_num)})

    # Atualizar linhas de `init_lab` a `end_lab` com `new_value` em 'label_measures'
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
    n = len(list_of_index)
    list_aux = []
    
    if n <= 1:  
        return list_of_index
    else: 
        # Select the first and second one on the similarity search
        list_aux.append(list_of_index.pop(0))
        list_aux.append(list_of_index.pop(0))

        result = find_close_values(list_aux, threshold)
        list_of_index.insert(0, result)
        
        return find_all_matches(list_of_index, threshold)
