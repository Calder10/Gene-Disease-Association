"""
Università degli Studi Di Palermo
Corso di Laurea Magistrale in Informatica
Anno Accademico 2019/2020
Elaborazione Dati - Modulo Big Data Management
Salvatore Calderaro 0704378
Email: salvatorecalderaro01@community.unipa.it
GENE-DESEASE ASSOCIATION ANALYZING SCIENTIFIC LITERATURE
"""


# Importo le librerie
from os import system
import csv
from pyspark.sql import SparkSession
import nltk
from Bio import Entrez
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SA
spark = SparkSession.builder.appName('gene_desease_association').getOrCreate()
Entrez.email="salvatorecalderaro01@community.unipa.it"

"""
Funzione che dato in input il nome del gene o il suo ID effettua una query
su Pubmed e restituisce Titolo e Abstract (se quest'ultimo è disponibile)
dei primi 100 articoli scientifici trovati.
"""

def find_papers(gene_id):
    query=Entrez.elink(dbfrom="pubmed", id=gene_id,linkname="gene_pubmed")
    result=Entrez.read(query)
    query.close()
    paper_id=[link["Id"] for link in result[0]["LinkSetDb"][0]["Link"]]
    #Considero i primi 100 articoli trovati
    """
    Controllare se è possibile selezionare gli articoli più rilevanti e con il
    maggior numero di citazioni.
    """
    # SELEZIONARE I PRIMI 200 ARTICOLI ED EVENTUALMENTE EFFETTUARE UNO SHUFFLE
    paper_id=paper_id[:10]
    c=1
    papers_list=[]
    for id_paper  in  paper_id:
        pubmed_entry = Entrez.efetch(db="pubmed", id=id_paper, retmode="xml")
        ris  = Entrez.read(pubmed_entry)
        article = ris['PubmedArticle'][0]['MedlineCitation']['Article']
        title=str(article['ArticleTitle'])
        """
        print("Articolo %d"%(c))
        print("Titolo:")
        print(article['ArticleTitle'])
        """
        if ('Abstract' in article):
            abstract=article['Abstract']['AbstractText']
            a=str(abstract[0])

            #print("Abstract:")
            #print(article['Abstract']['AbstractText'])
        else:
            a=""

        c=c+1
        r=(title,a)
        papers_list.append(r)
        #print("--------------------------------------------------------------------------")
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
            gene_info["taxname"] = ""
            continue
    gene_info.keys()
    #gene_info_list.append(gene_info)
    """
    print ("%s\t%s\t%s\t%s" % ("TaxonomyName","ID","OfficialSymbol","OfficialFullName"))
    print ("%s\t%s\t%s\t%s" % (gene_info_list[0]["taxname"],gene_info_list[0]["entrez_id"],gene_info_list[0]["official_symbol"],gene_info_list[0]["official_full_name
    """
    return (1,gene_info)


"""
Funzione che preso in input una serie di informazioni inerenti
il gene le memorizza all'interno di un dataframe
"""
def createGeneDataFrame(gene_info):
    val=[(str(gene_info['taxname']),str(gene_info['entrez_id']),str(gene_info['official_symbol']),str(gene_info['official_full_name']))]
    val
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
Funzione che preso in input l'ID del gene estrae le associazioni
gia note, fra quest'ultimo e le relative malattie dal database DisGenNet.
La funzione verrà utilizzata per verificare il grado di
confidenza dei risultati che restituirà in output il modello.
"""
def create_gene_desease_ass_from_DisGenNET(gene_id):
    f=open("data/all_gene_disease_associations.tsv")
    tsv_file=csv.reader(f,delimiter="\t")
    gene_des_ass=[]
    for row in tsv_file:
        x=(row[0].strip(' \t\n\r'))
        if(gene_id == x):
            t=str(row[0])
            t=(x,str(row[5]).strip(' \t\n\r'))
            gene_des_ass.append(t)
    df_ass=spark.createDataFrame(gene_des_ass,['ID Gene','Desease Name'])
    df_ass.show()
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
        if(t[1]=="NNS" or t[1]=="NN" or t[1]=="NNP" or t[1]=="FW" or t[1]=="SYM" or t[1]=="CD"):
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

system("clear")
(gene_id,info_gene)=init_data()
gene_df=createGeneDataFrame(info_gene)
papers_list=find_papers(gene_id)
paper_df=create_spark_dataframe(papers_list)
ass_df=create_gene_desease_ass_from_DisGenNET(gene_id)
clean_papers_df=clean_data(paper_df)
print_data_frame(clean_papers_df)
clean_papers_df=posTagging(clean_papers_df)
print_data_frame(clean_papers_df)
