"""
Università degli Studi Di Palermo
Corso di Laurea Magistrale in Informatica
Anno Accademico 2019/2020
Elaborazione Dati - Modulo Big Data Management
Salvatore Calderaro
GENE-DISEASE ASSOCIATION ANALYZING SCIENTIFIC LITERATURE
"""

# Importo le librerie
from os import system
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.functions import trim
from pyspark.sql import Row
from pyspark.sql.types import StringType
from pyspark import SparkContext
from pyspark.sql import SQLContext
import nltk
from Bio import Entrez
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import scispacy
import spacy
from spacy import displacy
from fuzzywuzzy import fuzz
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
ner = spacy.load("en_ner_bc5cdr_md")
spark = SparkSession.builder.appName('gene_desease_association').getOrCreate()
DisGenNET_path="data/all_gene_disease_associations.tsv"
res_fig_path="res/fig"
res_csv_path="res/csv"
med_term_path="data/med_term.csv"
Entrez.email="youremail@example.com"

"""
Funzione che dato in input il nome del gene o il suo ID effettua una query
su Pubmed e restituisce Titolo e Abstract (se quest'ultimo è disponibile)
dei primi 200 articoli scientifici trovati.
"""

def find_papers(gene_id):
    query=Entrez.elink(dbfrom="pubmed",sort="relevance", id=gene_id,linkname="gene_pubmed")
    result=Entrez.read(query)
    query.close()
    papers_list=[]
    try:
        paper_id=[link["Id"] for link in result[0]["LinkSetDb"][0]["Link"]]
    except IndexError:
        print("Spiacente non sono stati trovati articoli scientifici !")
        return papers_list

    print("Estrazione degli articoli in corso.....")
    #c=1
    paper_id=paper_id[:200]
    for id_paper  in  paper_id:
        #print(c)
        pubmed_entry = Entrez.efetch(db="pubmed", id=id_paper, retmode="xml")
        ris  = Entrez.read(pubmed_entry)
        try:
            article = ris['PubmedArticle'][0]['MedlineCitation']['Article']
            title=str(article['ArticleTitle'])
        except IndexError:
            title=""

        if ('Abstract' in article):
            try:
                abstract=article['Abstract']['AbstractText']
                a=str(abstract[0])
            except IndexError:
                a=""
        else:
            a=""
        r=(title,a)
        #c=c+1
        papers_list.append(r)
    print("Estazione articoli completata !")
    return papers_list

"""
Funzione che controlla se l'ID di un gene esiste e in caso positvo
restituisce le informaioni inerenti: l'Id, il nome,
il simbolo e la tassonomia
"""
def check_gene(gene_id):
    request = Entrez.epost("gene",id=gene_id)
    try:
        result = Entrez.read(request)
    except RuntimeError as e:
        return 0
    webEnv = result["WebEnv"]
    queryKey = result["QueryKey"]
    efetch_result = Entrez.efetch(db="gene", webenv=webEnv, query_key = queryKey, retmode="xml")
    gene = Entrez.read(efetch_result)
    gene_info={}
    gene_info_list = []
    gene=gene[0]
    try:
        gene_info["entrez_id"] = gene["Entrezgene_track-info"]["Gene-track"]["Gene-track_geneid"]
    except KeyError:
        gene_info["entrez_id"] = ""

    gene_info["official_symbol"] = "" # optional
    gene_info["official_full_name"] = "" # optional
    for gene_property in gene.get("Entrezgene_properties",[]):
        if gene_property.get("Gene-commentary_label",None) == "Nomenclature":
            for sub_property in gene_property["Gene-commentary_properties"]:
                if sub_property.get("Gene-commentary_label",None)  == "Official Symbol":
                    gene_info["official_symbol"] = sub_property.get("Gene-commentary_text","")
                if sub_property.get("Gene-commentary_label",None)  == "Official Full Name":
                    gene_info["official_full_name"] = sub_property.get("Gene-commentary_text","")
        try:
            gene_info["taxname"] = gene["Entrezgene_source"]["BioSource"]["BioSource_org"]["Org-ref"]["Org-ref_taxname"]
        except KeyError:
            gene_info["taxname"] = " "
            continue
    return (1,gene_info)

"""
Funzione che preso in input una serie di informazioni inerenti
il gene le memorizza all'interno di un dataframe
"""
def createGeneDataFrame(gene_info):
    try:
        val=[(str(gene_info['taxname']),str(gene_info['entrez_id']),str(gene_info['official_symbol']),str(gene_info['official_full_name']))]
    except KeyError:
        val=[(" ",str(gene_info['entrez_id']),str(gene_info['official_symbol']),str(gene_info['official_full_name']))]

    df_gene=spark.createDataFrame(val,["TaxonomyName","ID","OfficialSymbol","OfficialFullName"])
    df_gene.show(1,False)
    return df_gene

"""
Funzione che permetta all'utente di inserire da tastiera l'ID del gene
di cui devono essere trovate le malattie associate.
"""

def init_data():
    while (True):
        gene_id=input("Inserisci l'ID del gene--->")
        (f,info_gene)=check_gene(gene_id)
        if(f==1):
            break
        else:
            system("clear")
            print("L'ID %s non corrisponde ad alcun gene. Riprova!" %(gene_id))
    return (gene_id,info_gene)


"""
Funzione che presa in input una lista di tuple contenente
titolo ed abstract degli articoli li memorizza in un struttura
dati di tipo DataFrame.
"""
def create_spark_dataframe(papers):
    df_papers=spark.createDataFrame(papers,['Title','Abstract'])
    df_papers = df_papers.withColumn("Title", df_papers["Title"].cast(StringType()))
    df_papers = df_papers.withColumn("Abstract", df_papers["Abstract"].cast(StringType()))
    return df_papers

"""
Funzione che carica all'interno di un DataFrame
le associazioni fra geni e malattie memorizzate
all'interno del DisGenNET database.
"""

def loadDisGenNet():
    df = spark.read.csv(DisGenNET_path, sep=r'\t', header=True).select("geneId","diseaseName")
    df = df.withColumn("geneId", df["geneId"].cast(StringType()))
    df = df.withColumn('geneId', ltrim(df.geneId))
    return df

"""
Funzione che preso in input l'ID del gene estrae le associazioni
gia note, fra quest'ultimo e le relative malattie dal dataframe
in cui è stato caricato DisGenNET.
"""

def find_association_DisGenNET(df,gene_id):
    df_ass=df.filter(df.geneId == gene_id)
    return df_ass

"""
Funzione dato un testo elimina punteggiatura, stopwords ed
esegue la tokenizzazione.
"""
def remove_puntuaction_stop_words(text):
    #nltk.download("stopwords")
    en_stopwords=stopwords.words('english')
    tokenizer = RegexpTokenizer(r'\w+')
    tokens=tokenizer.tokenize(text)
    tokens_f=[]
    for token in tokens:
        if(token.lower() not in en_stopwords):
            tokens_f.append(token)

    return tokens_f

"""
Funzione che dato in input un dato testo
(già tokenizzato), ne esegue la lemmatizzaziome.
"""
def execute_lemmatization(tokens):
    lemmatizzer=WordNetLemmatizer()
    tokens_l=[]
    for token in tokens:
        t_l=lemmatizzer.lemmatize(token.lower())
        tokens_l.append(t_l)
    return tokens_l

"""
Funzione che preso in input il DataFrame con i papers
restituisce i papers dopo averne effettuato la pulitura.
(Rimozione della punteggiatura e delle stopwords, lemmatizzaziome
delle parole).
"""
def clean_data(paper_df):
    stop_words = set(stopwords.words('english'))
    clean_data=[]
    for row in  paper_df.rdd.collect():
        t=row['Title']
        a=row['Abstract']
        t_clean=remove_puntuaction_stop_words(t)
        a_clean=remove_puntuaction_stop_words(a)
        t_lemm=execute_lemmatization(t_clean)
        a_lemm=execute_lemmatization(a_clean)
        x=(t_lemm,a_lemm)
        clean_data.append(x)
    df_clean_papers=spark.createDataFrame(clean_data,['Title','Abstract'])
    return df_clean_papers


"""
Funzione che preso un testo su cui è stato già effettuato
il POS, restituisce una lista contenente solo i sostantivi
singolari e plurali, simboli e nomi propri.
"""

def remove_not_essentialPOS(tags):
    words=[]
    for t in tags:
        #or t[1]=="NNP"
        if(t[1]=="NNS" or t[1]=="NN" or t[1]=="FW" or t[1]=="SYM" or t[1]=="CD"):
            words.append(t[0])
    return words


"""
Funzione che preso in input un insieme di token effettua
il part of speech tagging.
La funzione restituirà un dataframe contenente solamente
le parole che vegono etichettate come sostantivi in
modo da ridurre la quantità di dati.
"""
def posTagging(clean_papers_df):
    #nltk.download('punkt')
    #nltk.download('averaged_perceptron_tagger')
    l=[]
    for row in  clean_papers_df.rdd.collect():
        t=row['Title']
        a=row['Abstract']
        t_tag=pos_tag(t)
        a_tag=pos_tag(a)
        t_tag_r=remove_not_essentialPOS(t_tag)
        a_tag_r=remove_not_essentialPOS(a_tag)
        x=(t_tag_r,a_tag_r)
        l.append(x)
    df_clean_papers=spark.createDataFrame(l,['Title','Abstract'])
    return df_clean_papers



"""
Funzione che scorrendo il dataframe in cui
sono presenti gli articoli applica per ciascuno
la NER. La funzione restituira in output un insieme
di malattie.
"""
def analyze_papers(clean_papers_df):
    diseases=[]
    print("Analisi letteratura scientifica in corso....")
    count=0
    for row in clean_papers_df.rdd.collect():
        t=row['Title']
        a=row['Abstract']
        l_t=lower_list(t)
        l_a=lower_list(a)
        x=l_t+l_a
        paper=" ".join(x)
        ris=apply_ner(paper)
        diseases+=ris
        count+=1
    print("Analisi completata !")
    return diseases

"""
Funziona che effettua la Named enity recognition
su di un testo, restituendo solo le entità
che vengono riconosciute come malattie.
"""

def apply_ner(text):
    diseases=[]
    doc=ner(text)
    #displacy.render(doc,style="ent")
    #print("______________________________________________________________________________________________________")
    for entity in doc.ents:
        if (entity.label_=="DISEASE" or entity.label_=="DESEASE"):
            if(len(str(entity))<=30):
                diseases.append(str(entity))
    return diseases


"""
Funzione che presa in input la lista delle malattie
ottenuta analizzando la letteratura scientifica
rimuove duplicati e quelle parole che per errore
potrebbero essere state identificate come malattie.
"""
def clean_diseases_list(diseases):
    clean_diseases = list(dict.fromkeys(diseases))
    for i in range(0,len(clean_diseases)):
        clean_diseases[i]=remove_duplicate_from_string(clean_diseases[i])
    clean_diseases=list(filter(filterWord,clean_diseases))
    return clean_diseases

"""
Funzione che preso in input l'insieme delle malattie
associate visualizza una WordCloud.
"""
def show_word_cloud(clean_diseases,gene_df,f):
    text=" ".join(clean_diseases)
    cloud=WordCloud(max_font_size=40).generate(text)
    row=gene_df.rdd.collect()
    if(f==0):
        title_fig="Malattie associate al gene " + str(row[0]['OfficialSymbol']) +"(" + str(row[0]['ID'])+")"
        path_fig=res_fig_path+"/"+ str(row[0]['OfficialSymbol']) +"(" + str(row[0]['ID'])+").png"
    else:
        title_fig="Malattie associate al gene " + str(row[0]['OfficialSymbol']) +"(" + str(row[0]['ID'])+")" + " filtrate"
        path_fig=res_fig_path+"/"+ str(row[0]['OfficialSymbol']) +"(" + str(row[0]['ID'])+")"+"_filter.png"
    #plt.figure(figsize=(20,8))
    plt.title(title_fig,fontsize=15)
    plt.imshow(cloud,interpolation="bilinear")
    plt.axis('off')
    cloud.to_file(path_fig)
    plt.show()

"""
Funzione che preso in inut il DataFrame con le malattie associate
al gene riversa i nomi in una lista.
"""
def create_diseases_list(ass_df):
    correct_disease_list=[]
    for row in ass_df.rdd.collect():
        d=str(row['diseaseName'])
        correct_disease_list.append(d)
    return correct_disease_list

"""
Funzione che preso in input un dataframe spark ne stampa il
contenuto.
"""

def print_data_frame(df):
    count=1
    for row in df.rdd.collect():
        print("\n %d" %(count))
        print(row['Title'])
        print(row['Abstract'])
        count+=1
        print("------------------------------------------------------------------------------")

"""
Funzione che converte tutti le stringhe di una lista
in minuscolo.
"""
def lower_list(l):
    for x in l:
        x.lower()
    return l

"""
Funzione che presa in input una stringa, elimina eventuali parole che occorrono
più volte.
"""
def remove_duplicate_from_string(text):
    t=text.split()
    clean_t = list(dict.fromkeys(t))
    clean_t=" ".join(clean_t)
    return clean_t

"""
Funzione che verifica se uns data parola è presente nelle parole da rimuovere.
Viene passata in input alla funzione filter.
"""
def filterWord(word):
    word_to_remove=upload_med_term()
    if word not in word_to_remove:
        return True
    else:
        return False

"""
Funzione che carica da un file CSV contenente una serie di termini medici che
poi serviranno per effettuare la ripulitura della lista di malattie.
"""
def upload_med_term():
    df = spark.read.csv(med_term_path,header=True).select("Name")
    df = df.withColumn("Name", df["Name"].cast(StringType()))
    row_list = df.select("Name").collect()
    med_term= [ row.Name for row in row_list]
    return med_term


"""
Funzione che presa in input una lista di malattie, la memorizza in un dataframe
e salva in memoria il corrispondente file CSV.
"""
def save_disease_toCSV(diseases,gene_df,f):
    row_gene=gene_df.rdd.collect()
    df = pd.DataFrame(diseases,columns =['DiseaseName'])

    if(f==0):
        filepath=res_csv_path+"/"+str(row_gene[0]['OfficialSymbol']) +"(" + str(row_gene[0]['ID'])+ ")_diseases.csv"
    else:
        filepath=res_csv_path+"/"+str(row_gene[0]['OfficialSymbol']) +"(" + str(row_gene[0]['ID'])+ ")_filtered_diseases.csv"
    df.to_csv(filepath, index = False)

"""
Funziona che confronta i risultati ottenuti con
le malattie associate al gene nel database DisGenNET.
Il confronto viene effettuato mediante fuzzy stringmatching
Viene inoltre calcolata la percentuale di malattie corrette rilevate.
"""
def evaluate_result(result, correct_result):
    tot=len(result)
    ris=[]
    for x in result:
        for y in correct_result:
            if x not in ris:
                score=fuzz.token_set_ratio(x,y)
                if(score >=80):
                    #print((x,y,score))
                    ris.append(x)
    matches_list = list(dict.fromkeys(ris))
    num_matches=len(matches_list)
    perc=(num_matches/tot)*100
    return(matches_list,perc)


"""
Main routine
"""
def main():
    system("clear")
    (gene_id,info_gene)=init_data()
    gene_df=createGeneDataFrame(info_gene)
    papers_list=find_papers(gene_id)
    if(len(papers_list)==0):
        sys.exit(1)
    else:
        paper_df=create_spark_dataframe(papers_list)
        paper_df.show(20)
        clean_papers_df=clean_data(paper_df)
        clean_papers_df.show(20)
        clean_papers_df=posTagging(clean_papers_df)
        clean_papers_df.show(20)
        diseases=analyze_papers(clean_papers_df)
        clean_diseases=clean_diseases_list(diseases)
        print("\n \n MALATTIE TROVATE ANALIZZANDO LA LETTERATURA SCIENTIFICA:")
        print(*clean_diseases, sep='\t')
        save_disease_toCSV(clean_diseases,gene_df,0)
        show_word_cloud(clean_diseases,gene_df,0)
        DisGenNET_df=loadDisGenNet()
        ass_df=find_association_DisGenNET(DisGenNET_df,gene_id)
        ass_df.show(20,False)
        correct_disease_list=create_diseases_list(ass_df)
        #print("LISTA DELLE MALATTIE ASSOCIATE AL GENE con ID: %s" %(gene_id))
        #print(*correct_disease_list, sep='\t')
        (final_result,perc)=evaluate_result(clean_diseases,correct_disease_list)
        print("\n \n DELLE MALATTIE IDENTIFICATE SOLO IL %.2f %% SONO RISULTATE CORRETTE:" %(perc))
        print(*final_result, sep='\t')
        save_disease_toCSV(final_result,gene_df,1)
        show_word_cloud(final_result,gene_df,1)
        exit(0)

if __name__ == "__main__":
    main()
