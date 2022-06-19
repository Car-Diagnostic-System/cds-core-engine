from unittest import TestCase
from main import *

class TestUnitDiagnose(TestCase):
    def test_download_s3_folder(self):
        actual_one = Diagnose.download_s3_folder('cds-bucket', 'pickles')
        actual_two = Diagnose.download_s3_folder('non-existed', 'pickles')
        self.assertEqual('download completed', actual_one)
        self.assertEqual('download failed', actual_two)