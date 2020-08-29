"""
Università degli Studi Di Palermo
Corso di Laurea Magistrale in Informatica
Anno Accademico 2019/2020
Elaborazione Dati - Modulo Big Data Management
Salvatore Calderaro 0704378
Email: salvatorecalderaro01@community.unipa.it
GENE-DISEASE ASSOCIATION ANALYZING SCIENTIFIC LITERATURE
"""

# Importo le librerie
from os import system
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.functions import trim
from pyspark.sql.types import StringType
import nltk
from Bio import Entrez
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from textblob import TextBlob
import scispacy
import spacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt
ner = spacy.load("en_ner_bc5cdr_md")
spark = SparkSession.builder.appName('gene_desease_association').getOrCreate()
DisGenNET_path="data/all_gene_disease_associations.tsv"
res_path="res"
Entrez.email="salvatorecalderaro01@community.unipa.it"

"""
Funzione che dato in input il nome del gene o il suo ID effettua una query
su Pubmed e restituisce Titolo e Abstract (se quest'ultimo è disponibile)
dei primi 100 articoli scientifici trovati.
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
    paper_id=paper_id[:200]
    for id_paper  in  paper_id:
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
        papers_list.append(r)
    print("Estazione articoli completata !")
    return papers_list

"""
Funzione che controlla se l'ID di un gene esiste e in caso positvo
stampa le informaioni inerenti: l'Id, il nome,
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
"""
def sentiment_papers(paper_df):
    for row in paper_df.rdd.collect():
        t=str(row['Title'])
        a=str(row['Abstract'])
        text=t + "\n" + a
        print(text)
        s=TextBlob(text)
        print(s.sentiment)
        print("******************************************************************")
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
    df_gene.show()
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
    df_papers.show(20)
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
contenente tutte le associazioni.
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
        t_l=lemmatizzer.lemmatize(token)
        tokens_l.append(t_l)
    return tokens_l

"""
Funzione che preso in input il DataFrame con i papers
restituisce i papers con la pulizia effettuata.
(Rimozione della punteggiatura e delle stopwords, lemmatizzaziome
delle parole.)
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
il POS, restituisce una lista contenente solo sostantivi
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
    print("Analisi completata ! Sono ststi analizzati %d articoli " %(count))
    return diseases

"""
Funziona che effettua la Named enity recognition
su di un testo, restituendo solo le entità
che vengono riconosciute come malattie.
"""

def apply_ner(text):
    diseases=[]
    doc=ner(text)
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
def show_word_cloud(clean_diseases,gene_df):
    text=" ".join(clean_diseases)
    cloud=WordCloud(background_color="white").generate(text)
    row=gene_df.rdd.collect()
    title_fig="Malattie associate al gene " + str(row[0]['OfficialSymbol']) +"(" + str(row[0]['ID'])+")"
    path_fig=res_path+"/"+ str(row[0]['OfficialSymbol']) +"(" + str(row[0]['ID'])+")"
    plt.figure(figsize=(20,8))
    plt.title(title_fig,fontsize=20)
    plt.imshow(cloud)
    plt.axis('off')
    plt.savefig(path_fig)
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
    word_to_remove=['infection',"disease","vaccine","heart","toxicity","stasis","shoulder","breast","drug","medicine","virus","head","inflammation","toxicity"]
    if word not in word_to_remove:
        return True
    else:
        return False

"""
Funzione che presa in input una lista ne stampa
il contenuto.
"""
def print_list(l):
    for x in l:
        print(x)
    print("______________________________________________________________________")


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
        DisGenNET_df=loadDisGenNet()
        ass_df=find_association_DisGenNET(DisGenNET_df,gene_id)
        ass_df.show(10)
        clean_papers_df=clean_data(paper_df)
        clean_papers_df=posTagging(clean_papers_df)
        diseases=analyze_papers(clean_papers_df)
        correct_disease_list=create_diseases_list(ass_df)
        clean_diseases=clean_diseases_list(diseases)
        print(correct_disease_list)
        print("\n \n MALATTIE TROVATE:")
        print(clean_diseases)
        #print_list(clean_diseases)
        show_word_cloud(clean_diseases,gene_df)
        exit(0)

main()
