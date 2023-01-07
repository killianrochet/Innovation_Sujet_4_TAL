# Rochet Killian
# Simoncelli Bastien
 
import pandas as pnd
import subprocess
import random
import time
import re
import unidecode
from scipy.spatial import distance
import numpy as np
 
 
# Pour Windows
#def fusion_file(cmd):
#    subprocess.run(["powershell", "-Command", cmd], capture_output=True)
 
# Execute une commande shell sur le système 
def use_command(cmd):
    os.system(cmd)
 
def merge_bigram(path):
    df = pnd.read_csv(path, sep="\t", skiprows=2, header=None)
    indexNumbers =  df[df[0] == '#--------------BIGRAMAS'].index
    df.drop(indexNumbers, inplace=True)
 
    new_df = df.groupby([0]).sum().sort_values(by=[1], ascending=False)
    new_df.to_csv('corpus.bi', sep="\t",)
 
# Permet la génération de phrase
def made_sentence(path, nbwords):
    df = pnd.read_csv(path, sep="\t", skiprows=2, header=None)
    # On random pour prendre le premier bigram qui commencera notre phrase
    sentence = random.choice(df[0])
    word_bool = table_asso(sentence.split(" ")[0], "^[V|N|A].*")
    while word_bool == False:
        sentence = random.choice(df[0])
        word_bool = table_asso(sentence.split(" ")[0], "^[V|N|A].*")
    count = 2
    start_word = sentence.split(" ")[1]
    end_bool = False
    # Tant que la phrase n'a pas au minimum le nombre voulu et que la phrase peut se terminer correctement
    while (count < nbwords or end_bool == False):
        # On cherche les N premiers bigrams qui commencent par "start_word"
        df_word = df[df[0].str.contains("^"+start_word+" ")].head(5)
        # On extrait au hasard le bigram parmis les N premiers (ici N = 5).
        new_bigram = df_word.sample()
        # On récupère tous les bigrams qui commence par "start_word"
        df_word_complete = df[df[0].str.contains("^"+start_word+" ")]
        # On sépare le bigram pour avoir d'un côté le premier mot et de l'autre le deuxième
        df_word = new_bigram[0].to_list()
        start_word = df_word[0].split(" ")[1]
        first_word = df_word[0].split(" ")[0]
        # appel de la fonction check_if_word_in_sentence
        start_word, boolean = check_if_word_in_sentence(sentence, df_word_complete, df_word, start_word)
        # On supprime le bigramme dans le df si il y est déjà une fois
        if boolean == True:
            indexNumbers = df[df[0] == first_word+" "+start_word].index
            df.drop(indexNumbers, inplace=True)
 
        sentence = sentence + " " + start_word
 
        if count >= nbwords:      
            # On va vérifier si on peut finir la phrase
            end_bool = table_asso(start_word, "^[V|C|D|P|R|Z|S|I].*")
            count == nbwords
 
        count += 1
        print(sentence)
 
# La fonction va vérifier si le mot (qui doit être de 4 caractères ou plus) est déjà présent dans la phrase
def check_if_word_in_sentence(sentence, df, bigram, second):
    boolean = False
    if " "+second+" " in sentence and len(second) >= 4:
        new_bigram = df[~df[0].str.contains(" "+second+"$")].head(1)
        bigram = new_bigram[0].to_list()
        second = bigram[0].split(" ")[1]
        boolean = True
    return second, boolean
 
# La fonction va s'assurer que le mot ne commence pas par le contenu de tagset
def table_asso(word, tagset):
    word_bool = True
    word_type = []
    fichier = open("TableAssociative", "r", encoding="utf8")
    for ligne in fichier:
        # On enlève les caractères spéciaux dans le mot, exemple à => a, é => e
        if "\t"+unidecode.unidecode(word)+"\t" in ligne:
            word_type.append(ligne.split(None, 1)[0])  # add only first word
        # On test avec le mot tel quel
        if "\t"+word+"\t" in ligne:
            word_type.append(ligne.split(None, 1)[0])  # add only first word
        # On test avec le mot tel quel et une majuscule sur la première lettre
        if "\t"+word.title()+"\t" in ligne:
            word_type.append(ligne.split(None, 1)[0])  # add only first word
    for item in word_type:
        if re.match(tagset, item):
            word_bool = False
    if word == "et":
        word_bool = False
    fichier.close()
    return word_bool
 
 
#####################
# PARTIE EMBEDDINGS #
#####################
 
# Prend en paramètre une phrase et retourne une phrase modifiée par rapport à une query
def change_sentence(sentence, query):
    tab_sentence = sentence.split()
    tab_tag_sentence = []
    tab_g_tag = []
    tab_word = []
 
    all_tag = extract_tagset()
 
    for i in range(len(tab_sentence)):
        tab_tag_word = find_tag_set(tab_sentence[i])
        tab_tag_sentence.append(tab_tag_word)
 
    for tag in tab_tag_sentence:
        tab_fl = extract_tag_set(tag)
        tab_g_tag.append(tab_fl)
 
    print(tab_tag_sentence)
    print(tab_g_tag)
    for g_tag in tab_g_tag:
        tab_t = []
        if len(g_tag) != 0:
            tab_t = find_word(g_tag)
        tab_word.append(tab_t)
 
    new_sentence = embeddings(tab_word, query, tab_sentence)
    print(sentence)
    return new_sentence
 
def embeddings(tab_word, query, tab_sentence):
    df = pnd.read_csv("embeddings-Fr.txt", sep="\t", header=None, skipinitialspace = True)
    my_emb = df[df[0] == query]
    # split chaque occurence pour en faire des pandas.series (il y en a 99)
    pandas_vec = my_emb[1].str.split(expand=True)
    # On passe d'un dataframe avec 99 pandas.series à un tuple (qui contient des str)
    emb_tuple = list(pandas_vec.itertuples(index=False, name=None))
    # on convertit les tuples qui sont en str en float pour appliquer la distance entre les vecteurs
    float_tuple = tuple(float(emb_float) for emb_float in emb_tuple[0])
    # on transforme en array
    float_tuple = np.array(float_tuple)
 
    cpt = 0
    new_sentence = ""
    for liste in tab_word:
        if len(liste) != 0:
            new_df = pnd.DataFrame(liste)
 
            # On récupère les lignes mot + vecteur si elles correspondent aux mots 
            final = df[df[0].isin(new_df[0])].dropna().reset_index(drop=True)
 
            # On supprime les mots déjà utilisés pour pas boucler dessus
            for word in new_sentence.split():
                indexNumbers = final[final[0] == word].index
                final.drop(indexNumbers, inplace=True)
 
            new_word = change_word(final, float_tuple, query)
        else: 
            new_word = tab_sentence[cpt]
        cpt += 1
        new_sentence = new_sentence + " " + new_word
 
    return new_sentence
 
 
def change_word(df, float_tuple, query):
    # On supprime les espaces inutiles qui vont empêcher de convertir en float
    df[1] = df[1].str.strip()
    df[1] = df[1].str.replace('    ',' ')
    df[1] = df[1].str.replace('   ',' ')
    df[1] = df[1].str.replace('  ',' ')
 
    cpt = 0
    df['dist'] = 100
 
    for sub in df[1]:
        res = tuple(map(float, sub.split(' ')))
        res = np.array(res)
        dist = distance.euclidean(float_tuple, res)
        #dist = np.linalg.norm(float_tuple - res)
        # On affecte la distance
        df.loc[cpt,'dist'] = dist
        cpt += 1
 
    # on récupère les 5 mots les plus proche
    print(df.sort_values(by = 'dist'))
    df_sort = df.sort_values(by = 'dist').head(5)
    # On randomise pour savoir quel mot on prend
    this_word = df_sort.sample()
 
    return this_word[0].to_string(index = False)
 
 
# Récupère tous les mots par rapport à un postag
def find_word(tab_tag):
    tab_word = []
    fichier = open("TableAssociative", "r", encoding="utf8")
    for ligne in fichier:
        for tag in tab_tag:
            if tag+"\t" in ligne:
                tab_word.extend(list(ligne.split(None, 1)[1].replace('\n','').split("\t")))
    return tab_word
 
# On extrait tous les Pos_tag"
def extract_tagset():
    tab_tagset = []
    fichier = open("TableAssociative", "r", encoding="utf8")
    for ligne in fichier:
        tab_tagset.append(ligne.split(None, 1)[0])
    return tab_tagset
 
# On extrait les tagset du mot
def extract_tag_set(tab_tagset):
    tab_word_tagset = []
    is_Ok = False
    for tagset in tab_tagset:
        if re.match('^NCC.*', tagset):
            # If word is a noun 
            tab_word_tagset.append(tagset)
        elif re.match('^A.*', tagset):
             # If word is an adjective 
            tab_word_tagset.append(tagset)
    return tab_word_tagset
 
 
# Prend en paramètre un mot et retourne ses tagsets
def find_tag_set(word):
    word_type = []
    fichier = open("TableAssociative", "r", encoding="utf8")
    for ligne in fichier:
        # On enlève les caractères spéciaux dans le mot, exemple à => a, é => e
        if "\t"+unidecode.unidecode(word)+"\t" in ligne:
            word_type.append(ligne.split(None, 1)[0])  # add only first word
        # On test avec le mot tel quel
        if "\t"+word+"\t" in ligne:
            word_type.append(ligne.split(None, 1)[0])  # add only first word
        # On test avec le mot tel quel et une majuscule sur la première lettre
        if "\t"+word.title()+"\t" in ligne:
            word_type.append(ligne.split(None, 1)[0])  # add only first word
    return word_type
 
 
def better_sentence(text):
    regex = r'(Le|La|la|le) '
    regex_voy = r'[aeiouAEIOUéèâyYh]'
 
    #Occ de regex dans la phrase
    matches = re.finditer(regex, text)
    for match in matches:
        # Mot qui suis la regex
        next_word_start = match.end()
        next_word_end = next_word_start + text[next_word_start:].index(" ")
        next_word = text[next_word_start:next_word_end]
        if re.match(regex_voy, next_word[0]): #If word commence par voyelle
            if match.start() > 1:
                text = text[:match.start()]+"l'"+text[match.end()-1:]
            else:
                text = text[:match.start()]+"L'"+text[match.end()-1:]
    return text
 
 
 
if __name__ == '__main__':
    # Temps 
    begin = time.perf_counter()
    # -- Lire le corpus
    #open_corpus("MEGALITE_FRANCAIS_bi/A/About,_Edmond-A_b_c_du_travailleur.pdf.seg.bi")
 
    ###########
    # Windows #
    ###########
 
    # -- Fusion des fichiers
    #fusion_command ="Get-Content .\MEGALITE_FRANCAIS_bi\*\*.bi | Out-File .\Fusion.bi"
 
    # -- Fusion de deux fichiers
    #fusion_command ="Get-Content .\MEGALITE_FRANCAIS_bi\I\*.bi | Out-File .\Fusion.bi -Encoding 'default'"
    #fusion_file(fusion_command)
 
    ########
    # UNIX #
    ########
 
    # -- Fusion des fichiers
    #fusion_command ="cat /MEGALITE_FRANCAIS_bi/*/*.bi > Fusion.bi"
    #use_command(fusion_command)
 
    # -- Enleve les lignes qui ne sont pas en UTF8
    #utf8_command ="grep -ax '.*' Fusion.bi > utf8Fusion.bi"
    #use_command(utf8_command)
 
    # -- Merge les fichiers pour enlever doublons
    #merge_bigram("utf8Fusion.bi")
 
    # -- Enlever les lignes avec le caractère ―
    #drop_line = "sed -i '/―/d' corpus.bi"
    #use_command(drop_line)
 
    # Enlever les caractères inutiles dans le fichier des embeddings
    #emb_command = "sed -i 's#[\[\]\,]##g' embeddings-Fr.txt" # vérifier pq ne fonctionne pas avec la regex
    #use_command(emb_command)
 
    # -- Permet de générer une phrase avec la méthode des bigrammes
    #made_sentence("corpus.bi", 8)
    #print(table_asso("et", "^[V|C|D|P|R|Z|S|I].*"))
 
    # -- Permet de générer une phrase avec les embeddings
    #gen_sentence_with_query(2,"embeddings-Fr.txt","horreur")
 
    # -- Permet de changer le sens de la phrase
    sentence = change_sentence("Un beau week-end","Horreur")
    print(sentence)
    #print(better_sentence(sentence))
 
    # temps d'exécution du programme
    end = time.perf_counter()
    print(f"Generate sentence in {end - begin:0.4f} seconds")