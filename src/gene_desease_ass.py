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
from os import  system
import csv
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('gene_desease_association').getOrCreate()
import nltk
from Bio import Entrez
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
nltk.download("vader_lexicon")
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SA
Entrez.email="salvatorecalderaro01@community.unipa.it"

"""
Funzione che dato in input il nome del gene o il suo ID effettua una query
su Pubmed e restituisce Titolo e Abstract
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
    paper_id=paper_id[:200]
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

        c=c+1
        r=(title,a)
        papers_list.append(r)
        #print("--------------------------------------------------------------------------")
    return papers_list

"""
Funzione che controlla se l'ID di un gene esiste e in caso positvo
stampa le informaioni inerenti: l'Id, il nome, il simbolo, la descrizione e la tassonomia
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

    gene_info_list.append(gene_info)
    print ("%s\t%s\t%s\t%s" % ("TaxonomyName","ID","OfficialSymbol","OfficialFullName"))
    print ("%s\t%s\t%s\t%s" % (gene_info_list[0]["taxname"],gene_info_list[0]["entrez_id"],gene_info_list[0]["official_symbol"],gene_info_list[0]["official_full_name"]))
    return 1

"""
Funzione che permetta all'utente di inserire da tastiera l'ID del gene
di cui devono essere trovate le malattie associate.
"""

def init_data():
    while (True):
        gene_id=input("Inserisci l'ID del gene--->")
        f=check_gene(gene_id)
        if(f==1):
            break
        else:
            system("clear")
            print("L'ID %s non corrisponde ad alcun gene. Riprova!" %(gene_id))
    return gene_id


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
Funzione che preso in input l'ID del gene estare le associazioni
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
    nltk.download("stopwords")
    en_stopwords=stopwords.words('english')
    tokenizer = RegexpTokenizer(r'\w+')
    tokens=tokenizer.tokenize(text)
    tokens
    #tokens = word_tokenize(text)
    tokens_f=[]
    for token in tokens:
        if(token.lower() not in en_stopwords):
            tokens_f.append(token)

    return tokens_f


"""
Funzione che preso in input il DataFrame con i papers
restituisce i papers con la pulizia effettuata.
(Rimozione della punteggiatura e delle stopwords)
"""
def clean_data(paper_df):
    stop_words = set(stopwords.words('english'))
    clean_data=[]
    for row in  paper_df.rdd.collect():
        t=row['Title']
        a=row['Abstract']
        t_clean=remove_puntuaction_stop_words(t)
        a_clean=remove_puntuaction_stop_words(a)
        x=(t_clean,a_clean)
        clean_data.append(x)
    df_clean_paoers=spark.createDataFrame(clean_data,['Title','Abstract'])
    return df_clean_paoers





gene_id=init_data()
papers_list=find_papers(gene_id)
paper_df=create_spark_dataframe(papers_list)
ass_df=create_gene_desease_ass_from_DisGenNET(gene_id)
clean_papers_df=clean_data(paper_df)

clean_papers_df.show()
