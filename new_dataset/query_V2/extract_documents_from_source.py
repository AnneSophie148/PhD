
import pandas as pd
import json
import string
import glob
import requests
import os
import urllib.parse
import xml.etree.ElementTree as ET
import config_APIs as config
from utils import *
import requests_html
import httpx

def build_doi_dict(dois_wos):
    '''Create a dictionary with the DOI prefix as key to search more easily for one DOI'''
    dic_dois = {}
    for doi_raw in dois_wos:
        if pd.isna(doi_raw):
            continue
        doi_str = normalize_doi(doi_raw)
        doi_prefix = doi_str.split("/")[0]
        dic_dois.setdefault(doi_prefix, []).append(doi_str)
    return dic_dois

def PLOS_extract_text(doi, file_name):
    '''Dowload both xml and pdf from PLOS dataset'''
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    xml_url = f"http://journals.plos.org/plosone/article/file?id={doi}&type=manuscript"
    pdf_url = f"http://journals.plos.org/plosone/article/file?id={doi}&type=printable"

    response_xml = requests.get(xml_url, timeout=10)
    if response_xml.status_code == 200 and response_xml.content.strip():

        xml_path = file_name + ".xml"
        with open(xml_path, 'wb') as f:
            f.write(response_xml.content)

    response_pdf = requests.get(pdf_url, timeout=10)
    if response_pdf.status_code == 200 and response_pdf.content.strip():
        pdf_path = file_name + ".pdf"
        with open(pdf_path, 'wb') as f:
            f.write(response_pdf.content)
        print(f"PDF saved from PLOS: {pdf_path}")
        return True
    else:
        return False

def Elsevier_extract_text(doi, file_name):
    '''Extract xml with Elsevier'''
    
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    API_key = config.APIKEY_ELSEVIER
    headers = {"X-ELS-APIKey": config.APIKEY_ELSEVIER, "X-ELS-Insttoken": config.ELSEVIER_INSTITUTIONAL_TOKEN, "Accept": "application/xml"}
    
    url = f"https://api.elsevier.com/content/article/doi/{doi}?view=FULL"
    timeout = httpx.Timeout(10.0, connect=60.0)
    client = httpx.Client(timeout=timeout, headers=headers)

    response = client.get(url)
    
    if response.status_code == 200:
        filename = file_name + ".xml"
        
        print(response.content)
        with open(filename, "wb") as f:
            f.write(response.content)
        print("\n")
        print(f"XML dowloaded with Elsevier {filename}")
        return True
    else:
        print(response.status_code)

        return False
        
def Wiley_extract_text(doi, file_name, max_retries=3, delay=10):
    #up to 3 articles per second, and 
    #up to 60 requests per 10 minutes, which entails building in a delay of 10 seconds between requests
    '''Extract pdf with Wiley API from a given DOI'''
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    API_key = config.APIKEY_WILEY
    headers = {"Wiley-TDM-Client-Token": API_key}

    proxies = {"http": f"http://{config.PROXY_USERNAME}:{config.PROXY_PASS}@{config.PROXY}",
    "https": f"http://{config.PROXY_USERNAME}:{config.PROXY_PASS}@{config.PROXY}"}


    url = f'https://api.wiley.com/onlinelibrary/tdm/v1/articles/{doi}'

    for attempt in range(1, max_retries + 1):
        #try again if fail because of code 500
        try:
            response = requests.get(url, headers=headers, proxies=proxies, timeout=10)

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return False

        if response.status_code == 200:
            filename = file_name + '.pdf'
            with open(filename, 'wb') as file:
                file.write(response.content)
            print(f'{filename} downloaded successfully with Wiley')
            return True

        elif 500 == response.status_code:
            print(f"Server error {response.status_code}. Retrying in {delay} seconds")
            print("Response text:", response.text)
            time.sleep(delay)
            continue

        else:
            return False

def download_pdf_from_page(page_url, file_path, link_selector=None, base_url=None, proxies=None, headers=None):
    """Download PDF from an HTML page given a selector for the link, or <meta name="citation_pdf_url">"""
    pdf_url = None

    response = requests.get(page_url, timeout=15, headers=headers, proxies=proxies)
    if response.status_code != 200:
        return False, page_url

    content_type = response.headers.get("Content-Type", "").lower()
    if "pdf" in content_type:
        with open(file_path, "wb") as f:
            f.write(response.content)
        return True, pdf_url

    soup = BeautifulSoup(response.content, "html.parser")

    #1. APS case: <a> tag
    if link_selector == "a":
        tag = soup.find("a", text="VIEW ARTICLE")
        if tag and tag.get("href"):
            pdf_url = tag["href"]

    #2. MicrobiologySoc case: <form>
    elif link_selector == "form":
        tag = soup.find("form", class_="ft-download-content__form")
        if tag and tag.get("action"):
            pdf_url = tag["action"]

    #3. Springer case: <meta name="citation_pdf_url">
    if not pdf_url:
        meta_tag = soup.find("meta", attrs={"name": "citation_pdf_url"})
        if meta_tag and meta_tag.get("content"):
            pdf_url = meta_tag["content"]

    if not pdf_url:
        return False, page_url

    if base_url:
        pdf_url = urljoin(base_url, pdf_url)
    else:
        pdf_url = urljoin(page_url, pdf_url)

    pdf_response = requests.get(pdf_url, timeout=15, headers=headers, proxies=proxies)
    if pdf_response.status_code == 200 and "pdf" in pdf_response.headers.get("Content-Type", "").lower():
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(pdf_response.content)
        return True, pdf_url

    return False, pdf_url

def retreive_microbilogy_soc(doi, file_name):
    pdf_path = file_name + ".pdf"
    headers = {'User-Agent': config.HEADER}

    microbiology_journals_variants = ["micro", "jgv", "ijsem", "acmi", "jmm", "mgen", "mout"]

    proxies = {"http": f"http://{config.PROXY_USERNAME}:{config.PROXY_PASS}@{config.PROXY}",
        "https": f"http://{config.PROXY_USERNAME}:{config.PROXY_PASS}@{config.PROXY}"}

    base_url = "https://www.microbiologyresearch.org"

    sigle_journal = doi.split("/")[-1].split(".")[0]

    #le sigle dans le doi est reconnu
    if sigle_journal in microbiology_journals_variants:
        pdf_url = f"{base_url}/content/journal/{sigle_journal}/{doi}"
        success, link = download_pdf_from_page(pdf_url, pdf_path, link_selector="form", base_url=base_url, headers=headers, proxies=proxies)
        if success:
            return True, link

    #sinon on teste toutes les variantes
    for var in microbiology_journals_variants:
        pdf_url = f"{base_url}/content/journal/{var}/{doi}"
        print(f"Trying var : {var} with link {pdf_url}")
        success, link = download_pdf_from_page(pdf_url, pdf_path, link_selector="form", base_url=base_url, headers=headers, proxies=proxies)
        if success:
            return True, link

    return False, None  

def retrieve_references_crossref(doi, file_name):
    '''Query crossref API to retreive OA location link if available and try to download, return true or false and link if false'''
    url = f"https://api.crossref.org/works/{doi}"
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    pdf_url = None
 
    headers = {'User-Agent': config.HEADER}
    
    proxies = {"http":f"http://{config.PROXY_USERNAME}:{config.PROXY_PASS}@{config.PROXY}",
            "https": f"http://{config.PROXY_USERNAME}:{config.PROXY_PASS}@{config.PROXY}"}

    response = requests.get(url, timeout=10, headers=headers, proxies=proxies)

    if response.status_code == 200:
        data = response.json()
        links = data["message"].get("link", [])
        pdf_url = links[0]["URL"] if links else None

        if "wiley" in pdf_url.lower() or "elsevier" in pdf_url.lower():
            #stop here if the document is in wiley or elsevier, we'll query the APIs in the next step
            return False, pdf_url

        if pdf_url:
            if "microbiologyresearch" in pdf_url:
                return retreive_microbilogy_soc(doi, file_name)

            if "www.apsnet.org/publications/" in pdf_url.lower():                
                success, link = download_pdf_from_page(pdf_url, pdf_path, link_selector="a", headers=headers, proxies=proxies)

            elif "springer" in pdf_url.lower():
                success, link = download_pdf_from_page(pdf_url, pdf_path, link_selector="meta", headers=headers, proxies=proxies)

        if success:
            return True, link

        pdf_path = file_name + ".pdf"

        try:
            pdf_response = requests.get(pdf_url, timeout=15)
            if pdf_response.status_code == 200:
                content_type = pdf_response.headers.get("Content-Type", "").lower()
                if "pdf" in content_type:
                    #avoids html
                    with open(pdf_path, "wb") as f:
                        f.write(pdf_response.content)
                    print(f"Dowloaded via crossref link : {pdf_path}")
                    return True

        except Exception as e:
            print(f"Error {e}")
            return False

    return False, pdf_url

def download_pdf_from_unpaywall(doi, file_name):
    '''Get link to pdf with unpaywall and try to download'''

    email = config.INSTITUTION_EMAIL
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    api_url = f"https://api.unpaywall.org/v2/{doi}?email={email}"

    headers = {"User-Agent": config.HEADER}
    proxies = {"http": f"http://{config.PROXY_USERNAME}:{config.PROXY_PASS}@{config.PROXY}",
        "https": f"http://{config.PROXY_USERNAME}:{config.PROXY_PASS}@{config.PROXY}"}


    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()

        oa_location = data.get("best_oa_location")
        if oa_location and oa_location.get("url_for_pdf"):
            pdf_url = oa_location["url_for_pdf"]
            if "wiley" in pdf_url.lower() or "elsevier" in pdf_url.lower():
                #stop here if the document is in wiley or elsevier, we'll query the APIs in the next step
                return False

            pdf_path = file_name + ".pdf"

            pdf_response = requests.get(pdf_url, timeout=10, headers=headers, proxies=proxies)
            pdf_response.raise_for_status()

            content_type = pdf_response.headers.get("Content-Type", "").lower()
            content_start = pdf_response.content[:100].lower()

            if "pdf" in content_type and b"<html" not in content_start and b"<!doctype html" not in content_start:
                #avoid html responds
                with open(pdf_path, "wb") as f:
                    f.write(pdf_response.content)
                print(f"PDF downloaded and saved to: {pdf_path}")
                return True, None
            else:
                #response is html content
                return False

        else:
            #no oa location
            return False

    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e}")
        return False

    except Exception as e:
        print(f"Error {e}")
        #any other error
        return False

def pmc_extract_paper(doi, file_name):
    '''Search for the document on PMC with the DOI then use the pmcid to request the full text'''
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:{doi}&format=json"

    response = requests.get(url)
    data = response.json()

    if data["hitCount"] > 0:
        result = data["resultList"]["result"][0]
        is_oa = result.get("isOpenAccess")

        pmcid = result.get("pmcid")
        if pmcid:
            xml_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
            xml_response = requests.get(xml_url)
            if xml_response.status_code == 200:
                xml_path = file_name + ".xml"
                print(xml_path)
                with open(xml_path, "wb") as f:
                    f.write(xml_response.content)
                print(f"Saved XML full text to: {xml_path}")
                return True
            else:
                print("Failed to download XML full text, status code:", xml_response.status_code)

    return False

def istex_extract_paper(doi, file_name):
    '''Search in ISTEX API'''
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    perso_token = config.TOKEN_ISTEX_API    
    #ADD " around the doi to avoid errors
    doi_encoded = urllib.parse.quote(f'"{doi}"')
    
    #same parameters as ISTEX demonstrator
    base_url = (f"https://api.istex.fr/document/?q=(doi%3A{doi_encoded})"
        "&facet=corpusName[*]&size=10&rankBy=qualityOverRelevance&output=*&stats")

    headers = {"Authorization": f"Bearer {perso_token}"}

    try:
        response = requests.get(base_url, headers=headers)
        response.raise_for_status()
        data = response.json()

        if data.get("total", 0) > 0:
            result = data["hits"][0]
            istex_id = result.get("id")
            print(istex_id)

            #get pdf, tei was old and sections were missing from the enrichment
            pdf_url = f"https://api.istex.fr/document/{istex_id}/fulltext/pdf"
            response = requests.get(pdf_url, headers=headers)
            response.raise_for_status()

            pdf_path = file_name + ".pdf"

            with open(pdf_path, "wb") as f:
                f.write(response.content)
            print(f"ISTEX PDF for {doi} saved")
            return True

    except Exception as e:
        print("Error querying ISTEX:", e)
    return False


def try_sources(doi, f_downloaded, f_not_downloaded, output_dir):
    '''All the sources we will query to extract the document, if no dowload save url and doi in file not_dowloaded'''
    
    sources = [
        ("PMC", pmc_extract_paper, {}),
        ("ISTEX", istex_extract_paper, {}),
        ("Unpaywall", download_pdf_from_unpaywall, {}),
        ("Crossref", retrieve_references_crossref, {})
    ]

    url_pdf = "none"  #if no URL is found

    for name, func, args in sources:
        local_dir = os.path.join(output_dir, name)
        os.makedirs(local_dir, exist_ok=True)

        file_name = os.path.join(local_dir, doi.replace('/', '_'))
        print("\n")
        print(f"Searching {doi} in {name}")
        try:
            result = func(doi=doi, file_name=file_name, **args)
        except Exception as e:
            print(f"Error while searching in {name}: {e}")
            result = False

        #for crossref that returns link
        if isinstance(result, tuple):
            #we could also get the url from unpaywall and not only crossref
            success, url_pdf_candidate = result
            if url_pdf_candidate:
                url_pdf = url_pdf_candidate

            if not success and url_pdf_candidate:
                print(f"{name} did not download {doi}, but found URL: {url_pdf}")
                
                #if crossref link contains elsevier or plos or wiley, query the according apis
                try:
                    if "elsevier" in url_pdf.lower():
                        print(f"Searching in Elsevier : {doi}")
                        if Elsevier_extract_text(doi, file_name=os.path.join(output_dir, f"Elsevier/{doi.replace('/', '_')}")):
                            f_downloaded.write(doi + "\n")
                            time.sleep(8)
                            return True
                    elif "plos" in url_pdf.lower():
                        print(f"Searching in PLOS : {doi}")
                        if PLOS_extract_text(doi, file_name=os.path.join(output_dir, f"PLOS/{doi.replace('/', '_')}")):
                            f_downloaded.write(doi + "\n")
                            return True
                       
                    elif "wiley" in url_pdf.lower():
                        if Wiley_extract_text(doi, file_name=os.path.join(output_dir, f"Wiley/{doi.replace('/', '_')}")):
                            print(f"Searching in Wiley : {doi}")
                            f_downloaded.write(doi + "\n")
                            #api wiley 10 secondes between each request necessary
                            #here 8 + 1*3 = 11 secs
                            time.sleep(8)
                            return True
                except Exception as e:
                    print(f"Error while trying secondary source for {doi}: {e}")

            result = success

        if result:
            f_downloaded.write(doi + "\n")
            return True
        time.sleep(1)

    # If none succeeded
    print("Not found in any source")
    f_not_downloaded.write(doi + " ; " + url_pdf + "\n")
    return False


def extract_docs(DOIs_WOS, Wos_dates, Wos_titles):
    """Dowload xml or pdf files with several APIs PMC, ISTEX, crossref, unpaywall, elsevier, willey, and PLOS dataset"""
    
    dic_dois = build_doi_dict(DOIs_WOS)

    #save dowloaded and non dowloaded dois
    #not_dowloaded file will also have the pdf link returned by crossref if available
    downloaded_file = "downloaded.lst"
    not_downloaded_file = "not_downloaded.lst"
    seen_dois = get_downloaded_dois()
    seen_dois = set(normalize_doi(doi).replace("doi-", "") for doi in seen_dois)

    import glob
    existing_downloads_citing = glob.glob("../query_articles/*/*")  #docs from all the sources
    doi_downloaded_list = [normalize_doi(os.path.splitext(os.path.basename(file))[0].replace("_", "/")) for file in existing_downloads_citing] #without extension xml or pdf

    print(len(seen_dois))

    #don't process filed for which the dowload did not work
    if os.path.exists(not_downloaded_file):
        with open(not_downloaded_file, "r") as f:
            for line in f:
                non_processed_doi = line.split(" ; ")[0].strip()
                seen_dois.add(non_processed_doi)


    with open(downloaded_file, "a") as f_downloaded, open(not_downloaded_file, "a") as f_not_downloaded:
        for key, doi_list in tqdm(dic_dois.items(), desc="Processing DOI prefixes"):
            for doi in tqdm(doi_list, desc=f"DOIs for prefix {key}", leave=False):
                if doi == "" or not doi:
                    print("Skip")
                    continue 

                if normalize_doi(doi) in seen_dois or normalize_doi(doi) in existing_downloads_citing:
                    print("Skip")
                    continue

                try_sources(doi, f_downloaded, f_not_downloaded, "../query_articles")

                            
        

if __name__ == "__main__":
    query_rslt = "references_WOS.tsv"
    df_WOS = pd.read_csv(query_rslt, sep="\t").fillna("")
    
    #Any type of document with Article --> We want to focus on research articles
    df = df_WOS[df_WOS["DT"].str.contains("Article", case=False, na=False)]
    DOIs_WOS = df["DI"].tolist()
    Wos_titles = df["TI"].tolist()
    Wos_dates = df["PY"].tolist()
    extract_docs(DOIs_WOS, Wos_dates, Wos_titles)
