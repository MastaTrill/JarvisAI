import sys

try:
    import pandas as pd
    import numpy as np
    from PIL import Image
    print('All imports successful!')
except Exception as e:
    print(f'Import error: {e}')

print('Python executable:', sys.executable)
