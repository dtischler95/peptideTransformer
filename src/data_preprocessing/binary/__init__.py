import os
import sys

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("""Package used for Preprocessing AMP data from most common AMP Data.
\nprepare_train_file should be able to process most of those databases if custom wish is needed,\n
 but it could be possible that you need to adjust the logic to your use-case. The other files are specific to this project""")