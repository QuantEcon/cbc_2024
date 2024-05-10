#! python
import os
import glob
dirs = '1_monday', '2_tuesday', '3_wednesday', '4_thursday', '5_friday'

print('Converting Myst files to notebooks.')
for dir in dirs:
    files = glob.glob(f'{dir}/*.md')
    for file in files:
        name, ext = file.split('.')
        print(f'Executing command jupytext --update --output {name}.ipynb {name}.md')
        os.system(f'jupytext --update --output {name}.ipynb {name}.md')
print('\nDone.')

