# Import Packages
import os
import pickle
import numpy as np

from Bio import Entrez
from Bio import Medline
from classes.DataHub import DataHub

# Allow PubMed access
Entrez.email = "marsenea@stanford.edu"

class MeshAnalyzer:

    def __init__ (self, pmids, mesh_dict_filename):
        self.pmids = pmids
        self.mesh_dict_filename = mesh_dict_filename

        self.mesh_dict = self._buildMeshDict()
        self.mesh_count_dict = self._buildMeshCount()
        self.sorted_mesh_terms = sorted(list(self.mesh_count_dict.keys()))
        self.document_mesh_matrix = self._buildDocumentMeshMatrix()

    # Private functions

    # Builds the document-MeSH matrix, which gives a one-hot encoding of MeSH terms for each document
    def _buildDocumentMeshMatrix(self):
        sorted_mesh_terms = self.sortMeshTerms()
        document_mesh_matrix = np.zeros((len(self.pmids), len(sorted_mesh_terms)))

        for doc_num, pmid in enumerate(self.pmids):
            for mesh_num, mesh in enumerate(sorted_mesh_terms):
                if mesh in self.mesh_dict[pmid]: document_mesh_matrix[doc_num][mesh_num] = 1
        self.document_mesh_matrix = document_mesh_matrix

        return document_mesh_matrix

    # Builds a dictionary mapping MeSH terms to the number of documents they appear with
    def _buildMeshCount(self):
        mesh_dict = self.mesh_dict
        mesh_count_dict = {}
        for pmid in mesh_dict.keys():
            for mesh_term in mesh_dict[pmid]:
                if mesh_term in mesh_count_dict.keys():
                    mesh_count_dict[mesh_term] += 1
                else:
                    mesh_count_dict[mesh_term] = 1
        return mesh_count_dict

    # Builds a dictionary mapping PMIDs to arrays of MeSH terms for the corresponding abstract
    def _buildMeshDict(self):
        if(os.path.exists(self.mesh_dict_filename + ".pkl")):
            pickle_in = open(self.mesh_dict_filename + ".pkl","rb")
            mesh_dict = pickle.load(pickle_in)
            return mesh_dict

        mesh_dict = {}

        handle = Entrez.efetch(db="pubmed", id=",".join(self.pmids), rettype="medline", retmode="text", usehistory="y")
        records = Medline.parse(handle)

        for record_num, record in enumerate(records):
            mesh_array = record.get("MH")
            for i in range(len(mesh_array)):
                mesh_array[i] = mesh_array[i].split('/')[0]
                if (mesh_array[i][0] == '*'): mesh_array[i] = mesh_array[i].split('*')[1]
                mesh_array[i] = mesh_array[i].split(',')[0]
            mesh_dict[self.pmids[record_num]] = mesh_array

        # Saves the MeSH dictionary in a pickle file
        with open(self.mesh_dict_filename + ".pkl", "wb") as f:
            pickle.dump(mesh_dict, f)

        return mesh_dict

    # Public functions

    # Filters out all MeSH terms paired with fewer than "threshold" documents
    def filterMesh(self, threshold):
        self.mesh_dict = {pmid: np.array([mesh for mesh in mesh_array if self.mesh_count_dict[mesh] >= threshold]) for (pmid, mesh_array) in self.mesh_dict.items()}
        self.mesh_count_dict = self._buildMeshCount()
        self.sorted_mesh_terms = sorted(list(self.mesh_count_dict.keys()))

    # Sorts MeSH terms into alphabetical array
    def sortMeshTerms(self):
        sorted_mesh_terms = sorted(list(self.mesh_count_dict.keys()))
        self.sorted_mesh_terms = sorted_mesh_terms
        return sorted_mesh_terms

    # Accessor functions

    def getDocumentMeshMatrix(self):
        return self.document_mesh_matrix

    def getMeshCount(self):
        return self.mesh_count_dict

    def getMeshDict(self):
        return self.mesh_dict

    def getNumMesh(self):
        return len(self.mesh_count_dict.keys())
