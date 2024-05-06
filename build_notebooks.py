#! python
import os
import glob
dirs = 'monday', 'tuesday', 'wednesday', 'thursday', 'friday'

print('Converting Myst files to notebooks.')
for dir in dirs:
    files = glob.glob(f'{dir}/*.md')
    for file in files:
        name, ext = file.split('.')
        os.system(f'jupytext --update --output {name}.ipynb {name}.md')
print('\nDone.')

