from unittest import TestCase
from main import *

class TestUnitDiagnose(TestCase):
    def test_download_s3_folder(self):
        Diagnose.download_s3_folder('cds-bucket', 'pickles')
        self.assertEqual(1, 1)