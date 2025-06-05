from AU.AUExtraction import extract_au
import os

AU_output = os.path.join(os.getcwd(), "data", "AU_output_rev")
apex_output = os.path.join(os.getcwd(), "data", "apex_output")

print(AU_output, apex_output)
extract_au(apex_output, AU_output)