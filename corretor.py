# UNIVERSIDADE ESTADUAL DE CAMPINAS
# FT - UNICAMP / CURSO SISTEMAS DE INFORMAÇÃO
#TCC - CORRETOR ORTOGRÁFICO PARA ANALISE DE TIPOS DE ERROS E SUA RELAÇÃO COM O ANALFABETISMO
# AUTOR - MATHEUS EDUARDO DA SILVA RA: 230719
# ORIENTADORA -ANA ESTELA ANTUNES DA SILVA

import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup 
import nltk
#nltk.download('punkt')
import os
import pandas as pd 
from nltk.test.portuguese_en_fixt import setup_module
import nltk.corpus
import re
from collections import Counter

#split_words(): recebe uma lista de strings, e cria um novo vetor que contem somente palavras válidas(sem caracteres nao alfanumericos). 
def split_words(tokens_list):
    list_words = []
    for token in tokens_list:
        if token.isalpha():
            list_words.append(token)
    return list_words

#normalization_words(): normalizando as palavras(tornando todas em minúsculas).
def normalization_words(list_words):
    normalized_list = []
    for word in list_words:
        normalized_list.append(word.lower())
    return normalized_list


#Corretor Ortográfico:
#known_words(): retorna a lista de palavras e sua classificação de erro caso existam no dicionário.
def known_words(words): 
    lista = set(w for w in words if w[0] in normalized_list_dict)
    return lista

#candidates_words(): geração de palavras candidatas a possibilidades de correção.
def candidates_words(word):
    candidates = known_words([word]) | known_words(edits_distance_1(word)) | set([word])
    return candidates

#probability():Probabilidade da palavra ser a correte dentre as possíveis.
def probability(word): 
    if isinstance(word, tuple) == True:
        return (frequency[word[0]] /words_total)
    else:
        return (frequency[word]/words_total)

#correction(): função base do corretor, recebe uma palavra e retorna sua maior probabilidade de correção da palavra.
def correction(word):
    result = max(candidates_words(word), key=probability)
    if isinstance(result, tuple) != True:
        return (word, "")
    else:
        return (result)
    
#Possiblidades de correção de palavras:
    #Grupos de erros(classificação):
    #1- Emprego das consoantes e dos dígrafos + Emprego das formas que representam o som nasal
    #2- Emprego de vogais
    #3- Acréscimo e omissão de letras
    #4- Inversão de letras
    #5- Letras com formato semelhante
    #6- Erros decorrentes de escritas particulares + Segmentação indevida das palavras
    #7- Uso de acentuação

#edit_word_insert(): edição de palavras através de adição.
def edit_word_insert(word_sliced, letters):
    error_group = 3
    error_group_content = []
    new_possible_words = []
    for E, D in word_sliced:
        for letter in letters:
            new_possible_words.append(E + letter + D)
            error_group_content.append(error_group)
    words = list(zip(new_possible_words, error_group_content))
    return words

#edit_word_delete(): edição de palavras através de remoção.
def edit_word_delete(word_sliced, letters):
    error_group = 3
    error_group_content = []
    new_possible_words = []
    for E, D in word_sliced:
        if len(D) > 0:
            new_possible_words.append(E + D[1:])
            error_group_content.append(error_group)
    words = list(zip(new_possible_words, error_group_content))
    return words

#edit_word_transpose(): edição de palavras através de transposição.
def edit_word_transpose(word_sliced, letters):
    error_group = 4
    error_group_content = []
    new_possible_words = []
    for E, D in word_sliced:
        if len(D) > 1:
            new_possible_words.append(E + D[1] + D[0] + D[2:])
            error_group_content.append(error_group)
    words = list(zip(new_possible_words, error_group_content))
    return words

#edit_word_replace(): edição de palavras através de substituição.
def edit_word_replace(word_sliced, letters):
    error_group = 0 
    error_group_content = []
    new_possible_words = []
    vogals = ['a', 'e', 'i', 'o', 'u']
    digrafos = ['ch', 'lh', 'nh', 'rr', 'ss', 'sc', 'sç', 'xc', 'xs', 'am', 'an', 'em', 'en', 'im', 'in', 'om', 'on', 'um', 'un', 'gu', 'qu'] 
    letters_equals_p = ['p', 'f', 'q']
    letters_equals_m = ['m', 'n']
    accents = ['á', 'â', 'à', 'ã', 'é', 'ê', 'è', 'ẽ', 'í', 'î', 'ì', 'ĩ', 'ó', 'ô', 'õ', 'ò', 'ú', 'û', 'ù', 'ũ', 'ç']
    for E, D in word_sliced:
        if len(D) > 0:
            for letter in letters:
                removed_letter = D[:1]
                new_possible_words.append(E + letter + D[1:])
                if len(E) > 1 and len(D) > 1:
                    possible_digraphs = [(E[-1] + letter), (letter + D[1])]
                    next_letter_digraph = D[1:2]
                elif len(D) > 1 and len(E) <= 1:
                    possible_digraphs = [(E + letter), (letter + D[1])]
                    next_letter_digraph = D[1:2]
                elif len(E) > 1 and len(D) == 1:
                    possible_digraphs = [(E[-1] + letter), (letter)]
                    next_letter_digraph = ""
                else:
                    possible_digraphs = [(E + letter), (letter)]
                    next_letter_digraph = ""
                if ((possible_digraphs[0] in digrafos) or (possible_digraphs[1] in digrafos)) \
                      and (next_letter_digraph not in vogals and next_letter_digraph != 'h')  and (letter != removed_letter): 
                    error_group = 1 #digrafos, consoantes ou sons nasais
                elif (((removed_letter in letters_equals_p) and (letter in letters_equals_p)) \
                      or ((removed_letter in letters_equals_m) and (letter in letters_equals_m))) and (letter != removed_letter): 
                    error_group = 5
                elif ((((removed_letter not in vogals) and (removed_letter not in accents)) \
                        and ((letter not in vogals) and (letter not in accents)) and (removed_letter not in accents))) and (letter != removed_letter):
                    error_group = 1 #digrafos, consoantes ou sons nasais
                elif ((removed_letter in vogals) and (letter in vogals)) and (letter != removed_letter): #vogais incorretas
                    error_group = 2 #uso de vogais
                elif (((removed_letter in vogals) and (letter in accents)) or (removed_letter == 'c' and letter == 'ç')): #erros de acentuacao
                    error_group = 7 #erro de acentuação
                else:
                    error_group = 6 #erros de escritas particulares
                error_group_content.append(error_group)
    words = list(zip(new_possible_words, error_group_content))
    return words 

#edits_distance_1(): execução das correções na distância de edição 1 para as palavras no corretor.
def edits_distance_1(word):
    word_sliced = []
    letters = 'abcdefghijklmnopqrstuvwxyzáâàãéêèẽíîìĩóôõòúûùũç'
    for i in range(len(word)+1):
        word_sliced.append((word[:i], word[i:]))

    deletes    = edit_word_delete(word_sliced, letters)
    inserts    = edit_word_insert(word_sliced, letters)
    transposes = edit_word_transpose(word_sliced, letters)
    replaces   = edit_word_replace(word_sliced, letters)
    return set(deletes + transposes + replaces + inserts)



#criador de dados de teste:
def create_test_dataset(file):
    list_words_test = []
    correct = ''
    wrong = ''
    error_group = 0
    f = open(file, "r", encoding='UTF-8')
    for line in f:
        correct, wrong, error_group = line.split(" ", 3)
        list_words_test.append((correct, wrong, error_group))
    f.close()
    return list_words_test

#ajustar formato das saidas do corretor:
def adjust_format(word):
    if isinstance(word, tuple) != True:
        return (word, "" )
    else:
        return word

#avaliador do corretor ortográfico: ( corrigir para mostrar as porcentagens de tipo de erro)
def evalutate_corrections(tests, vocabulary):
    number_words = len(tests)
    right_corrections_right_group = 0
    right_corrections_wrong_group = 0
    wrong_corrections_right_group = 0
    wrong_corrections_wrong_group = 0
    list_RR = []
    list_RW = []
    list_WR = []
    list_WW = []
    list_unknown = []
    unknown = 0
    for correct, wrong, error_group in tests:
        error_group = error_group[0]
        corrected_word = correction(wrong)
        corrected_word = adjust_format(corrected_word)
        result = [corrected_word[0], corrected_word[1], correct, error_group]
        if (correct not in vocabulary):
            list_unknown.append(result)
            unknown += 1
        elif (str(corrected_word[0]) == correct) and (str(corrected_word[1]) == str(error_group)):
            list_RR.append(result)
            right_corrections_right_group += 1 
        elif (str(corrected_word[0]) == correct) and (str(corrected_word[1]) != str(error_group)):
            list_RW.append(result)
            right_corrections_wrong_group += 1 
        elif (str(corrected_word[0]) != correct) and (str(corrected_word[1]) == str(error_group)):
            list_WR.append(result)
            wrong_corrections_right_group += 1
        else:
            list_WW.append(result)
            wrong_corrections_wrong_group += 1
    accuracy_percentage = round(right_corrections_right_group*100/number_words, 2)
    accuracy_percentage_error_group = round((right_corrections_right_group + wrong_corrections_right_group)*100/number_words, 2)
    unknown_percentage = round(unknown*100/number_words, 2)
    print('Grupo WR : ' , list_WR)
    print('Grupo RW : ' , list_RW)
    print('Grupo WW : ', list_WW)
    print('\n')
    print(f"{accuracy_percentage}% de {number_words} palavras, desconhecida é {unknown_percentage}%,  acurácia de erros de grupo {accuracy_percentage_error_group}%")
    print("Palavras Desconhecidas: " + str(len(list_unknown)))
    print("Grupo palavras WW: " + str(len(list_WW)))
    print("Grupo palavras WR: " + str(len(list_WR)))
    print("Grupo palavras RW: " + str(len(list_RW)))
    print("Grupo palavras RR: " + str(len(list_RR)))
    return list_RR, list_RW, list_WR, list_WW, list_unknown

#main()
if __name__ == '__main__':
    #registrando palavaras do dicionario em ptbr
    with open('dicionario_USP_ptBr.txt', encoding='UTF-8') as f:
        dict_content = f.read()
    dict_content = nltk.tokenize.word_tokenize(dict_content)
    #criando vocabulario e dicionario de palavras
    folha_sp = nltk.corpus.mac_morpho.words()
    list_words_dict = split_words(dict_content)
    list_words_voc_common_words_folhasp = split_words(folha_sp)
    print(f"Número de palavras(dicionario PTBR) é {len(list_words_dict)}")
    print(f"Número de palavras(vocabulario Folha SP) é {len(list_words_voc_common_words_folhasp)}")
    normalized_list_dict = normalization_words(list_words_dict)
    normalized_list_voc_folhaSP = normalization_words(list_words_voc_common_words_folhasp)
    #calculando a frequencia das palavras e total de palavras no vocabulario
    frequency = nltk.FreqDist(normalized_list_voc_folhaSP)
    words_total = len(normalized_list_voc_folhaSP)
    #executando testes com o corretor ortografico e o corpora de teste
    words_testing_list = create_test_dataset('words_testing.txt')
    list_RR, list_RW, list_WR, list_WW, list_unknown = evalutate_corrections(words_testing_list, normalized_list_voc_folhaSP)
