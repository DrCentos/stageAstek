from dataclasses import dataclass

table_schema = 'operateur:STRING, annee:INTEGER, filiere:STRING, code_categorie_consommation:STRING, libelle_categorie_consommation:STRING, code_grand_secteur:STRING, libelle_grand_secteur:STRING, code_naf:INTEGER, libelle_secteur_naf2: STRING, conso:FLOAT, pdl:INTEGER, indqual:FLOAT, nombre_mailles_secretisees:INTEGER, code_region:INTEGER, libelle_region:STRING, naf_code:STRING, naf_INTEGER_vf:STRING, naf_INTEGER_v2_65c:STRING, naf_INTEGER_v2_40c: STRING'

@dataclass
class Conso:
    operateur: str
    annee: int
    filiere: str
    code_categorie_consommation: str
    libelle_categorie_consommation: str
    code_grand_secteur: str
    libelle_grand_secteur: str
    code_naf: int
    libelle_secteur_naf2: str
    conso: float
    pdl: int
    indqual: float
    nombre_mailles_secretisees: int
    code_region: int
    libelle_region: str

@dataclass
class Naf:
    ligne: int
    code: str
    int_vf: str
    int_v2_65c: str
    int_v2_40c: str

@dataclass
class ConsoAndNaf:
    operateur: str
    annee: int
    filiere: str
    code_categorie_consommation: str
    libelle_categorie_consommation: str
    code_grand_secteur: str
    libelle_grand_secteur: str
    code_naf: int
    libelle_secteur_naf2: str
    conso: float
    pdl: int
    indqual: float
    nombre_mailles_secretisees: int
    code_region: int
    libelle_region: str
    naf_code: str
    naf_int_vf: str
    naf_int_v2_65c: str
    naf_int_v2_40c: str

