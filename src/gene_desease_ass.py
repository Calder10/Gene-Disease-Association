"""
Università degli Studi Di Palermo
Corso di Laurea Magistrale in Informatica
Anno Accademico 2019/2020
Elaborazione Dati - Modulo Big Data Management
Salvatore Calderaro 0704378
GENE-DESEASE ASSOCIATION ANALYZING SCIENTIFIC LITERATURE
"""
# Importo le librerie
from os import  system
from pyspark.sql import SparkSession
from Bio import Entrez
Entrez.email="salvatorecalderaro01@community.unipa.it"

"""
Funzione che dato in input il nome del gene o il suo ID effettua una query
su Pubmed e restituisce Titolo e Abstract (o eventuali parti
salienti dell'articolo qualora fosse disponibile)
dei primi 20 articoli scientifici più rilevanti.
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
    paper_id=paper_id[:100]
    c=1
    for id_paper  in  paper_id:
        pubmed_entry = Entrez.efetch(db="pubmed", id=id_paper, retmode="xml")
        ris  = Entrez.read(pubmed_entry)
        article = ris['PubmedArticle'][0]['MedlineCitation']['Article']
        print("Articolo %d"%(c))
        print("Titolo:")
        print(article['ArticleTitle'])
        if ('Abstract' in article):
            print("Abstract:")
            print(article['Abstract']['AbstractText'])
        c=c+1
        print("--------------------------------------------------------------------------")

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
L'utente può anche inserire il nome del gene e il sistema si occupera
di andare a ricavare l'ID associato per potere effettuare la query.
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

gene_id=init_data()
print("Stampo i titoli dei papers inerenti questo gene")
find_papers(gene_id)
