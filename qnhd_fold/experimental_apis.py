# experimental_apis.py
# Real API integration for PDB, UniProt, and AlphaFold DB

import urllib.request
import json
from typing import Dict

class ExperimentalDataAPI:
    '''Integration with experimental databases'''
    
    def __init__(self):
        self.pdb_api = "https://data.rcsb.org/rest/v1/core/entry/"
        self.uniprot_api = "https://rest.uniprot.org/uniprotkb/"
        self.alphafold_api = "https://alphafold.ebi.ac.uk/api/prediction/"
    
    def fetch_pdb_structure(self, pdb_id: str) -> Dict:
        '''Fetch experimental structure from RCSB PDB'''
        url = f"{self.pdb_api}{pdb_id}"
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode())
        return {
            'pdb_id': pdb_id,
            'title': data.get('struct', {}).get('title', ''),
            'method': data.get('exptl', [{}])[0].get('method', ''),
            'resolution': data.get('rcsb_entry_info', {}).get('resolution_combined', [None])[0],
            'organism': data.get('rcsb_entity_source_organism', [{}])[0].get('ncbi_scientific_name', '')
        }
    
    def fetch_uniprot_entry(self, uniprot_id: str) -> Dict:
        '''Fetch protein from UniProt'''
        url = f"{self.uniprot_api}{uniprot_id}.json"
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode())
        return {
            'uniprot_id': uniprot_id,
            'protein_name': data.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', ''),
            'sequence': data.get('sequence', {}).get('value', ''),
            'length': data.get('sequence', {}).get('length', 0)
        }
