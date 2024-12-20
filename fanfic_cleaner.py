import re    
import os
import json
import tqdm

def process_file(path_to_file):
    FIELDS = ['Category', 
            'Genre', 
            'Language', 
            'Status', 
            'Published', 
            'Updated', 
            'Packaged', 
            'Rating', 
            'Chapters', 
            'Words', 
            'Publisher', 
            'Story URL', 
            'Author URL', 
            'Summary']

    with open(path_to_file,'r') as f:
        retrieve_metadata = True
        full_text = []
        metadata = {k: None for k in FIELDS}
        for line in f:
            if retrieve_metadata:
                field_check = True
                for field in FIELDS:
                    if line.startswith(field):
                        field_check = False
                        metadata[field] = line.replace(f'{field}: ', '').strip()
                        if field == 'Summary':
                            retrieve_metadata = False
                if line != '\n' and field_check:
                    if line.startswith('by '):
                        metadata['Author'] = line.replace('by ', '').strip()
                    else:
                        metadata['Title'] = line.strip()

            else:
                full_text.append(line)
            
    return metadata, ''.join(full_text)

ROOT = 'data/fanfic/fanfictiondotnet_repack/'
DEST = 'data/fanfic/texts/'
META_DEST = 'data/fanfic/metadata.jsonl'

all_walk = list(os.walk(ROOT))
with open(META_DEST, 'w') as f:
    for root, dirs, files in tqdm.tqdm(all_walk):
        if len(files) > 0:
            for file in files:
                if file.endswith('.txt'):
                    path_to_file = os.path.join(root, file)
                    metadata, full_text = process_file(path_to_file)
                    metadata['File'] = file
                    f.write(json.dumps(metadata) + '\n')
                    with open(os.path.join(DEST, file), 'w') as g:
                        g.write(full_text.strip())
