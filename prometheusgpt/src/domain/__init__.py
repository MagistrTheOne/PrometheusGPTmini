# PrometheusGPT Mini - Domain Specialization Package
# Author: MagistrTheOne, Krasnodar, 2025

from .military_finetuner import MilitaryDocumentFineTuner
from .medical_finetuner import MedicalTextFineTuner
from .financial_finetuner import FinancialReportFineTuner

__all__ = [
    'MilitaryDocumentFineTuner',
    'MedicalTextFineTuner', 
    'FinancialReportFineTuner'
]
